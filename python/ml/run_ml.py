import math
import os
import pickle
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import pearsonr
from torch import Tensor

from constants import MNT_PATH
import ml.dft_predictor_nn
import ml.exact_cluster_expansion
import ml.classification_nn
import ml.simple_nn
import remote.database
import studies.spinel.spinel as spinel
from ml.fit_data import FitData, GPRFitData, Metric
from ml.kernels import DFTKernel
from ml.scaler import Scaler
from remote.node import Node

DEVICE = 'cuda'
MIN_ADORBATES_PER_MAT = 65
ALL_ADORBATES_PER_REG_MAT = 108
STRIDES_50 = [100003, 100019, 100043, 100049, 100057, 100069, 100103, 100109, 100129, 100151, 100153, 100169,
               100183, 100189, 100193, 100207, 100213, 100237, 100267, 100271, 100279, 100291, 100297, 100313,
               100333, 100343, 100357, 100361, 100363, 100379, 100391, 100393, 100403, 100411, 100417, 100447,
               100459, 100469, 100483, 100493, 100501, 100511, 100517, 100519, 100523, 100537, 100547, 100549,
               100559, 100591, 100609]
STRIDES_10 = STRIDES_50[:10]
STRIDES_1 = STRIDES_50[:1]


async def fetch_adsorbate_Von_training_data(node: Node, T_K: float, n_steps: int,
                                            low_entropy_only: bool) -> tuple[Tensor, Tensor, Tensor]:
    X = []
    Y = []
    barriers_eV = []

    studies_with_converged = await remote.database.fetch_steps(node, T_K, n_steps)
    for study_data in studies_with_converged:
        if len(study_data.adsorbates) < MIN_ADORBATES_PER_MAT:
            continue

        if low_entropy_only and 'O8' in study_data.material:
            continue

        s = spinel.spinel_from_name(study_data.material, '100o')

        Eref = None
        for adsorbate, step in study_data.adsorbates.items():
            if adsorbate == 'clean':
                Eref = step.Efree0V_eV
                break
        else:
            continue

        for adsorbate, step in study_data.adsorbates.items():
            if adsorbate == 'clean':
                continue
            encoding = s.adsorbate_full_encoding(adsorbate, simple_only=False)
            X.append(encoding)
            Y.append((step.Efree0V_eV - Eref, len(barriers_eV), step.index))

        barriers_eV.append(compute_barrier_eV(n_steps, study_data))

    return torch.stack(X), torch.tensor(Y, dtype=torch.double), torch.tensor(barriers_eV, dtype=torch.double)

async def fetch_min_adsorbate_training_data(node: Node, T_K: float, n_steps: int, simple_only: bool,
                                            ) -> tuple[Tensor, Tensor, list[int]]:
    X = []
    Y = []
    ad_lookup = [[] for _ in range(n_steps)]
    ad_enc_lookup = [[] for _ in range(n_steps)]
    initialized_ad_lookup = False

    studies_with_converged = await remote.database.fetch_steps(node, T_K, n_steps)
    for study_data in studies_with_converged:
        if len(study_data.adsorbates) != ALL_ADORBATES_PER_REG_MAT:
            continue

        s = spinel.spinel_from_name(study_data.material, '100o')
        if simple_only and not s.is_simple:
            continue

        if not initialized_ad_lookup:
            for adsorbate, step in study_data.adsorbates.items():
                ad_lookup[step.index].append(adsorbate)
                O_tensor, oct_tensor = s.encode_adsorbate(adsorbate)
                ad_tensor = torch.cat([O_tensor, oct_tensor])
                ad_enc_lookup[step.index].append(ad_tensor)
            initialized_ad_lookup = True


        Eref = None
        for adsorbate, step in study_data.adsorbates.items():
            if adsorbate == 'clean':
                Eref = step.Efree0V_eV
                break
        else:
            raise ValueError('No clean step found')

        step_Es = [len(ad_lookup[i]) * [None] for i in range(n_steps)]
        for adsorbate, step in study_data.adsorbates.items():
            adsorbate_index = ad_lookup[step.index].index(adsorbate)
            step_Es[step.index][adsorbate_index] = step.Efree0V_eV - Eref
        all_Es = [Ei for i in range(n_steps) for Ei in step_Es[i]]
        
        X.append(s.one_hot_encoding(simple_only))
        # jahn_teller_distortion = max(study_data.lattice_abc) / min(study_data.lattice_abc) - 1
        # combined_encoding = torch.cat([s.one_hot_encoding, torch.tensor([jahn_teller_distortion], dtype=torch.double)])
        # combined_encoding = torch.cat([s.one_hot_encoding, torch.tensor(study_data.magnetic_moments[:2], dtype=torch.double)])
        # o0, _, o2, _, o4, _, o6, o7, _, _, _, _, _, _ = study_data.oxidation_states
        # combined_encoding = torch.cat([s.one_hot_encoding, torch.tensor([o0, o2, o4, o6, o7], dtype=torch.double)])
        # combined_encoding = torch.cat([s.one_hot_encoding, torch.tensor(study_data.lattice_abc, dtype=torch.double) / 10])
        # X.append(combined_encoding)
        Y.append(torch.tensor(all_Es, dtype=torch.double))
    
    ad_encs = torch.stack([enc for i in range(n_steps) for enc in ad_enc_lookup[i]])

    step_sizes = [len(names) for names in ad_lookup]
    if sum(step_sizes) != len(Y[0]):
        raise ValueError('Mismatch between step sizes and number of adsorbates')
    return torch.stack(X), torch.stack(Y), step_sizes, ad_encs


async def fetch_ratio_Von_training_data(node: Node, T_K: float, n_steps: int,
                                        low_entropy_only: bool) -> tuple[Tensor, Tensor]:
    ratio_map: dict[tuple[int, ...],
                    tuple[remote.database.StudyData, float]] = {}

    studies_with_converged = await remote.database.fetch_steps(node, T_K, n_steps)
    for study_data in studies_with_converged:
        if len(study_data.adsorbates) < MIN_ADORBATES_PER_MAT:
            continue

        if len(study_data.adsorbates) != ALL_ADORBATES_PER_REG_MAT:
            continue

        if low_entropy_only and 'O8' in study_data.material:
            continue

        s = spinel.spinel_from_name(study_data.material, '100o')
        if s.AABBBB[2] != s.AABBBB[3] or s.AABBBB[4] != s.AABBBB[5]:
            continue

        barrier_eV = compute_barrier_eV(n_steps, study_data)
        ratio_encoding = tuple(s.one_hot_encoding.tolist())
        if (ratio_encoding in ratio_map and ratio_map[ratio_encoding][1] > barrier_eV) or ratio_encoding not in ratio_map:
            ratio_map[ratio_encoding] = (study_data, barrier_eV)

    X = []
    Y = []
    for ratio_encoding, (study_data, dE) in ratio_map.items():
        X.append(torch.tensor(ratio_encoding, dtype=torch.double))
        Y.append(dE)

    return torch.stack(X), torch.tensor(Y, dtype=torch.double)


def compute_barrier_eV(n_steps, study_data):
    lowest_Es = torch.tensor(n_steps * [float('inf')], dtype=torch.double)
    for step in study_data.adsorbates.values():
        lowest_Es[step.index] = min(lowest_Es[step.index], step.Efree0V_eV)
    barrier_eV = max(lowest_Es - lowest_Es.roll(1))
    return barrier_eV


def compute_Y_barriers_eV(n_steps, Y):
    Y_barriers_eV = []
    for i in range(int(max(Y[:, 1])) + 1):
        Y_i = Y[Y[:, 1] == i][:, 0]
        step_i = Y[Y[:, 1] == i][:, 2]
        lowest_Es = torch.tensor(n_steps * [float('inf')], dtype=torch.double)
        for Y_ij, step_ij in zip(Y_i, step_i):
            step_ij = int(step_ij)
            lowest_Es[step_ij] = min(lowest_Es[step_ij], Y_ij)
        Y_barriers_eV.append(max(lowest_Es - lowest_Es.roll(1)))
    return torch.tensor(Y_barriers_eV, dtype=torch.double)


def format_step(fit: FitData, time_s: float) -> str:
    parts = [f't(s):{time_s:6.1f}', f'RMAE_train={int(fit.R_train*100)}|{fit.MAE_train:.3f}']
    if fit.MAE_test is not None and math.isfinite(fit.MAE_test):
        R_test = int(
            fit.R_test*100) if fit.R_test is not None and math.isfinite(fit.R_test) else fit.R_test
        parts += [f'RMAE_test={R_test}|{fit.MAE_test:.3f}']
    return '  '.join(parts)


# Adam 0.001 ro 0.0001
# Try LBFGS for the mll since its not stochaistic
# ADd noise to the diagonal and add likelihood noise to the likelihood to solve PSD issue
def train_with_regularization(X: Tensor, Y: Tensor, scaler_X: Scaler, scaler_Y: Scaler,
                              kernel: DFTKernel, norm_power: float, norm_lambda: float, *,
                              max_iter: int = 51, lr: float = 0.001, history_size: int = 10,
                              time_start: float | None = None) -> SingleTaskGP:
    time_start = time_start or time.time()
    print(f'\nStarting L{norm_power} norm with λ={norm_lambda:.5f}  t(s):{time.time() - time_start:6.1f}')
    X_normalized = scaler_X.transform(X)
    Y_nornalized = scaler_Y.transform(Y)
    model = SingleTaskGP(train_X=X_normalized, train_Y=Y_nornalized, covar_module=kernel).to(DEVICE)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Replace Adam with LBFGS
    optimizer = torch.optim.LBFGS(mll.parameters(), lr=lr, max_iter=max_iter, history_size=history_size)

    def closure():
        optimizer.zero_grad()
        output = model(X_normalized)  # pylint: disable=not-callable
        loss = -mll(output, model.train_targets)
        # norm_value = model.covar_module.L_norm(norm_power)
        # loss += norm_lambda * norm_value
        loss.backward()
        return loss

    for i in range(max_iter):
        # Step with LBFGS optimizer
        optimizer.step(closure)

        if i % 5 == 0:
            model.eval()
            fit = GPRFitData.create(model, X, Y, scaler_X, scaler_Y)
            model.train()
            print(f'{i:5d}  Nsig={fit.N_params:3d}  RMAE={int(fit.R_train*100)}|{fit.MAE_train:.7f}')

    model.eval()
    return model


def optimize_sigmas(X: Tensor, Y: Tensor, scaler_X: Scaler, scaler_Y: Scaler, X_test: Tensor,
                    Y_test: Tensor, n: int, depth: int) -> FitData:
    time_start = time.time()
    start_model = train_with_regularization(X, Y, scaler_X, scaler_Y, DFTKernel(n, depth), 0, 0, time_start=time_start)
    fit = GPRFitData.create(start_model, X, Y, scaler_X,
                            scaler_Y, X_test=X_test, Y_test=Y_test)
    print(f'  {format_step(fit, time.time() - time_start)}')
    return fit


def get_kfold_split_masks(N: int, k_fold: int, stride: int | None, X: Tensor) -> list[Tensor]:
    
    if stride is None:
        # The first spinel.OHE_LENGTH columns are the one-hot encoding of the material in X
        unique_materials = X[:, :spinel.OHE_LENGTH].unique(dim=0)  # Get unique one-hot encoded materials
        
        material_masks = []
        for material in unique_materials:
            material_mask = (X[:, :spinel.OHE_LENGTH] != material).any(dim=1)  # Mask out one material at a time
            material_masks.append(material_mask.cpu())

        # Randomly partition material masks into k_fold groups
        random.shuffle(material_masks)
        masks = [torch.ones(N, dtype=bool) for _ in range(k_fold)]
        for i, material_mask in enumerate(material_masks):
            masks[i % k_fold] &= material_mask  # Logical & of all masks within a group
        return masks
    
    else:
        # Generate the pseudorandomly shuffled indices
        shuffled_indices = [(j * stride) % N for j in range(N)]

        # Split the shuffled indices into k_fold_size chunks
        k_fold_size = N // k_fold
        chunks = [shuffled_indices[i * k_fold_size: (i + 1) * k_fold_size] for i in range(k_fold)]

        # Distribute any leftover elements evenly across the chunks
        for i, idx in enumerate(shuffled_indices[k_fold * k_fold_size:]):
            chunks[i % len(chunks)].append(idx)

        masks = []
        for chunk in chunks:
            mask = torch.ones(N, dtype=bool)
            for i in chunk:
                mask[i] = False
            masks.append(mask)
        return masks



def print_test_misses(fit: FitData, names: tuple[str, ...], encodings: Tensor, n: int = 20):
    if fit.Yhat_test_mean is None:
        print("No test data provided.")
        return

    named_misses = list(zip(names, fit.Y_test - fit.Yhat_test_mean,
                        fit.Y_test, fit.Yhat_test_mean, encodings))
    print(' Miss   True   Pred  Name            Encoding')
    misses_sorted = sorted(named_misses, key=lambda x: -abs(x[1]))
    for i in range(min(n, len(misses_sorted))):
        name, miss, Y_true, Y_predicted, encoding = misses_sorted[i]
        print(f'{miss.item():5.2f}  {Y_true:5.2f}  {Y_predicted:5.2f}  {name:14}  {[int(e) for e in encoding]}')


def cluster_expansion_kfold(params, X, Y, k_fold, s):
    params = torch.Tensor(params)
    group_lengths = [5, 5, 5, 4, 8]
    pair_mask = ml.exact_cluster_expansion.make_pair_mask(group_lengths).to(DEVICE)
    non_zero_pair_indices = torch.nonzero(pair_mask)

    triplet_mask = ml.exact_cluster_expansion.make_triplet_mask(group_lengths).to(DEVICE)
    non_zero_triplet_indices = torch.nonzero(triplet_mask)

    quad_mask = ml.exact_cluster_expansion.make_quad_mask(group_lengths).to(DEVICE)
    non_zero_quad_indices = torch.nonzero(quad_mask)

    assert (len(non_zero_pair_indices) + len(non_zero_triplet_indices) + len(non_zero_quad_indices) == len(params))

    pair_inclusion = {int(x) for [x] in torch.nonzero(params[:len(non_zero_pair_indices)])}
    pair_indices = [x for i, x in enumerate(non_zero_pair_indices) if i in pair_inclusion]
    pair_indices = torch.stack(pair_indices)

    triplet_inclusion = {int(x) for [x] in torch.nonzero(
        params[len(non_zero_pair_indices):len(non_zero_pair_indices) + len(non_zero_triplet_indices)])}
    triplet_indices = [x for i, x in enumerate(non_zero_triplet_indices) if i in triplet_inclusion]
    triplet_indices = torch.stack(triplet_indices)

    quad_inclusion = {int(x) for [x] in torch.nonzero(
        params[len(non_zero_pair_indices) + len(non_zero_triplet_indices):])}
    quad_indices = [x for i, x in enumerate(non_zero_quad_indices) if i in quad_inclusion]
    quad_indices = torch.stack(quad_indices)
    kwargs = {'pair_indices': pair_indices, 'triplet_indices': triplet_indices,
              'quad_indices': quad_indices, 'ridge_lambda': 1e-6}
    metric = kfold_cross_validation(X, Y, k_fold, s, kwargs, 'cluster_expansion', verbosity=0)
    return metric


def kfold_cross_validation(X: Tensor, Y: Tensor, k_fold: int, stride: int, kwargs: dict[str, Any],
                           training_type: str, *, verbosity: int = 3) -> Metric:
    N = X.shape[0]
    if stride and math.gcd(N, stride) != 1:
        raise ValueError(f'N={N} and stride={stride} must be coprime.')
    if verbosity >= 2:
        print(f'Starting k-fold cross validation for encoding with N={N}')

    k_fold_fits: list[FitData] = []
    for mask in get_kfold_split_masks(N, k_fold, stride, X):
        # fit = optimize_sigmas(X[mask], Y[mask], scaler_X, scaler_Y, X[~mask], Y[~mask], X.shape[1], X.shape[1])
        if training_type == 'ratio':
            fit = ml.simple_nn.train_and_predict(X[mask], Y[mask], X[~mask], Y[~mask], verbosity=verbosity, **kwargs)
        elif training_type == 'intermediate_energy':
            fit = ml.dft_predictor_nn.train_and_predict(
                X[mask], Y[mask], X[~mask], Y[~mask], verbosity=verbosity, **kwargs)
        else:
            raise NotImplementedError(f'Unknown training type: {training_type}')

        k_fold_fits.append(fit)
        if verbosity >= 3:
            print(format_step(fit, 0))

    Yhat_test_mean = torch.cat([fit.Yhat_test_mean for fit in k_fold_fits])
    Y_test = torch.cat([fit.Y_test for fit in k_fold_fits])
    if training_type == 'ratio':
        R_test = pearsonr(Y_test.cpu().numpy(), Yhat_test_mean.cpu().numpy())[0]
        MAE_test = (Y_test - Yhat_test_mean).abs().mean().item()
    elif training_type == 'intermediate_energy':
        R_test = pearsonr(Y_test[:, 0].cpu().numpy(), Yhat_test_mean[:, 0].cpu().numpy())[0]
        MAE_test = (Y_test[:, 0] - Yhat_test_mean[:, 0]).abs().mean().item()
    else:
        raise NotImplementedError(f'Unknown training type: {training_type}')

    metric = Metric(None, Yhat_test_mean, Y_test, MAE_test, R_test)

    if verbosity >= 1:
        print(f'Finished  N:{N}  k-fold:{k_fold}  stride:{stride}  R{metric.R_test:.2f}  MAE:{metric.MAE_test:.3f}')

    return metric


def kfold_CV_min_adsorbate(X: Tensor, Y: Tensor, k_fold: int, stride: int, kwargs: dict[str, Any],
                           *, verbosity: int = 3) -> list[Metric]:
    N = X.shape[0]
    if stride and math.gcd(N, stride) != 1:
        raise ValueError(f'N={N} and stride={stride} must be coprime.')
    if verbosity >= 2:
        print(f'Starting k-fold cross validation for encoding with N={N}')

    all_fits = [ml.classification_nn.train_and_predict(X[mask], Y[mask], X[~mask], Y[~mask], verbosity=verbosity, **kwargs) for mask in get_kfold_split_masks(N, k_fold, stride, X)]

    metrics = []
    for k_fold_fits in zip(*all_fits):
        Yhat_test_mean = torch.cat([fit.Yhat_test_mean for fit in k_fold_fits])
        Y_test = torch.cat([fit.Y_test for fit in k_fold_fits])
        R_test = pearsonr(Y_test.cpu().numpy(), Yhat_test_mean.cpu().numpy())[0]
        MAE_test = (Y_test - Yhat_test_mean).abs().mean().item()

        metric = Metric(None, Yhat_test_mean, Y_test, MAE_test, R_test)
        if verbosity >= 1:
            print(f'Finished  N:{N}  k-fold:{k_fold}  stride:{stride}  R{metric.R_test:.2f}  MAE:{metric.MAE_test:.3f}')
        metrics.append(metric)

    return metrics


async def train_dft_mep_prediction(
        node: Node,
        T_K: float,
        *,
        k_fold: int | None = 10,
        acq_func: str = 'EI',
        training_type: str = 'intermediate_energy',
        low_entropy_only: bool = False,
        write_results: bool = False,
):
    min_dEs = None
    kwargs = {}
    if training_type == 'ratio':
        X, Y = await fetch_ratio_Von_training_data(node, T_K, 4, low_entropy_only)
        Ymu_MAE = (Y - Y.mean()).abs().mean().item()
        x0 = [[0.04, 0.2]]
        gamma = 0.995
        hidden_dims = (32,)
        strides = STRIDES_50
    elif training_type == 'min_adsorbate':
        X, Y, step_sizes, Yenc = await fetch_min_adsorbate_training_data(node, T_K, 4, low_entropy_only)
        Ymu_MAE = 0
        kwargs['step_sizes'] = step_sizes
        kwargs['Yenc'] = Yenc
        kwargs['N_samples'] = 12
        x0= [[x * 1e-5, y] for y in [0.004, 0.005, 0.006, 0.007] for x in [6, 7, 8, 9, 10]]
        # x0 = [[5e-5, 0.04]]
        gamma = 0.999
        hidden_dims =(512,)
        strides = STRIDES_50
    elif training_type == 'intermediate_energy':
        X, Y, min_dEs = await fetch_adsorbate_Von_training_data(node, T_K, 4, low_entropy_only)
        Ymu_MAE = (min_dEs - min_dEs.mean()).abs().mean().item()
        # x0 = [[0.0001, 0.0227668], [0.001, 0.01]]
        x0 = [[x*1e-3, y*1e-3] for x in [5, 10, 20, 40, 80] for y in [1, 2, 4, 8]]
        gamma = 0.95
        hidden_dims = (256, 256)
        strides = STRIDES_1
    else:
        raise ValueError(f'Unknown training type: {training_type}')

    X = X.to(device=DEVICE, dtype=torch.float32)
    Y = Y.to(device=DEVICE, dtype=torch.float32)
    print(f'Training using {acq_func} with N={X.shape[0]} hl={hidden_dims} splits={len(strides)} k_fold={k_fold}')
    
    def objective(params):
        metrics: list[Metric] = []

        # Display 0% initially
        sys.stdout.write(f"\rCalculating {len(strides)} x {k_fold}-Fold CVs... 0%")
        sys.stdout.flush()

        for i, s in enumerate(strides):
            decay, lr = params
            full_kwargs = kwargs | {'hidden_dims': hidden_dims,'decay': decay, 'lr': lr, 'gamma': gamma}
            metric = kfold_cross_validation(X, Y, k_fold, s, full_kwargs, training_type, verbosity=0)
            metrics.append(metric)

            # Update the percentage display
            percent = int(100 * (i + 1) / len(strides))
            sys.stdout.write(f"\rCalculating {len(strides)} x {k_fold}-Fold CVs... {percent}%")
            sys.stdout.flush()
        sorted_metrics = sorted(metrics, key=lambda metric: metric.R_test)
        median_metric = sorted_metrics[len(strides)//2]
        mean_MAE = np.mean([metric.MAE_test for metric in metrics])
        mean_R2 = np.mean([metric.R_test**2 for metric in metrics])

        # Clear the loading text
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        # Print the results
        print_str = '['
        print_str += f' {decay:.6f}, {lr:.6f}, {gamma:.5f},'
        if training_type == 'intermediate_energy':
            Yhat_dEs = compute_Y_barriers_eV(4, median_metric.Yhat_test_mean)
            R_Yhat_dEs = pearsonr(min_dEs.numpy(), Yhat_dEs.numpy())[0]
            MAE_Yhat_dEs = (min_dEs - Yhat_dEs).abs().mean().item()
            print_str += f' {MAE_Yhat_dEs:.5f}, {R_Yhat_dEs**2:.5f},'
        print_str += f' {mean_MAE:.5f}, {mean_R2:.5f}, {Ymu_MAE:.5f}],'
        print(print_str)

        if write_results:
            if training_type == 'ratio' or training_type == 'min_adsorbate':
                print(median_metric.Yhat_test_mean.tolist())
                print(median_metric.Y_test.tolist())
            elif training_type == 'intermediate_energy':
                filename = 'low_entropy_intermediate_energies' if low_entropy_only else 'intermediate_energies'
                with open(os.path.join(MNT_PATH, 'pickles', f'{filename}.pkl'), 'wb') as f:
                    pickle.dump((median_metric.Y_test[:, 0].tolist(), median_metric.Yhat_test_mean[:, 0].tolist()), f)
                filename = 'low_entropy_dEs' if low_entropy_only else 'all_dEs'
                with open(os.path.join(MNT_PATH, 'pickles', f'{filename}.pkl'), 'wb') as f:
                    pickle.dump((min_dEs.tolist(), Yhat_dEs.tolist()), f)
            else:
                raise NotImplementedError(f'Unknown training type: {training_type}')

        # return MAE_Yhat_dEs
        return mean_MAE

    for param in x0:
        objective(param)


async def train_dft_mep_partial_eval(
        node: Node,
        T_K: float,
        *,
        k_fold: int = 10,
        simple_only: bool = True,
):
    X, Y, step_sizes, Yenc = await fetch_min_adsorbate_training_data(node, T_K, 4, simple_only)
    X = X.to(device=DEVICE, dtype=torch.float32)
    Y = Y.to(device=DEVICE, dtype=torch.float32)
    gammax1000 = 995
    decay, lr, gamma = [0.00008, 0.005, gammax1000 / 1000.0]
    hidden_dims =(512,)
    random_strides = 20 * STRIDES_50
    strides = STRIDES_10
    print(f'Training with N={X.shape[0]} hl={hidden_dims} splits={len(strides)} k_fold={k_fold}'
          ' decay={decay:.5f} lr={lr:.7f} gamma={gamma:.5f}')

    def predict_Von(randomized: bool) -> tuple[list[float], list[float]]:
        all_metrics: list[list[Metric]] = []
        current_strides = random_strides if randomized else strides

        # Display 0% initially
        sys.stdout.write(f"\rCalculating {len(current_strides)} x {k_fold}-Fold CVs... 0%")
        sys.stdout.flush()

        for i, s in enumerate(current_strides):
            kwargs = {'hidden_dims': hidden_dims,'decay': decay, 'lr': lr, 'gamma': gamma,
                      'randomized': randomized, 'step_sizes': step_sizes, 'Yenc': Yenc}
            all_metrics.append(kfold_CV_min_adsorbate(X, Y, k_fold, s, kwargs, verbosity=0))

            # Update the percentage display
            percent = int(100 * (i + 1) / len(current_strides))
            sys.stdout.write(f"\rCalculating {len(current_strides)} x {k_fold}-Fold CVs... {percent}%")
            sys.stdout.flush()
        
        # Clear the loading text
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

        MAEs_mean = []
        for metrics in zip(*all_metrics):
            mean_MAE = np.mean([metric.MAE_test for metric in metrics])
            print(f'[{decay:.5f}, {lr:.7f}, {gamma:.5f}, {mean_MAE:.5f}],')
            MAEs_mean.append(mean_MAE)
        return list(range(1, max(step_sizes) + 1)), MAEs_mean

    MAEs_random = list(zip(*predict_Von(True)))
    print()
    MAEs_predicted = list(zip(*predict_Von(False)))

    if Yenc is None:
        filename = 'min_k_predict_Von'
    else:
        filename = f'min_k_predict_Von_Yenc_{gammax1000}'

    pickle_path = os.path.join(MNT_PATH, 'pickles', f'{filename}.pkl')
    print(f'Writing results to {pickle_path}')
    with open(pickle_path, 'wb') as f:
        pickle.dump((MAEs_predicted, MAEs_random), f)
