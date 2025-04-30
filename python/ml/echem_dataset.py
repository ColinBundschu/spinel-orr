import dataclasses
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch import Tensor

from ml.adsorbates_encoding import AdsorbatesEncoding


class Encoding(Enum):
    INDEX = 'index encoding'
    OHE = 'one hot encoding with task indices'
    PROP = 'integer based properties encoding'
    NAME = 'adsorbate name encoding'


@dataclass(frozen=True)
class EchemDataset:
    X_i: Tensor
    X_ohe: Tensor
    X_prop: Tensor
    X_name: tuple[str, ...]
    Y: Tensor
    encoding: AdsorbatesEncoding
    Ecycle_eV: float
    _: dataclasses.KW_ONLY
    Vons_V: Tensor = None  # Set by this class in __post_init__
    best_ad_indices: Tensor = None  # Set by this class in __post_init__

    def __post_init__(self):
        if not self.X_ohe.shape[0] == self.X_prop.shape[0] == self.X_i.shape[0] == self.Y.shape[0] == len(self.X_name):
            raise ValueError('Lengths of X and Y do not match')

        Vons_V, best_ad_indices = Von_single(self.Y, self.encoding.step_mask, self.Ecycle_eV)
        object.__setattr__(self, 'Vons_V', Vons_V)
        object.__setattr__(self, 'best_ad_indices', best_ad_indices)

    @property
    def N(self) -> float:
        '''Number of data points in the entire domain X'''
        return self.X_ohe.shape[0]

    @property
    def d_ohe(self) -> float:
        '''Number of features in the domain X_ohe'''
        return self.X_ohe.shape[1]

    @property
    def d_prop(self) -> float:
        '''Number of features in the domain X_prop'''
        return self.X_prop.shape[1]

    def X_encoded(self, encoding: Encoding) -> Tensor:
        match encoding:
            case Encoding.INDEX:
                return self.X_i
            case Encoding.OHE:
                return self.X_ohe
            case Encoding.PROP:
                return self.X_prop
            case Encoding.NAME:
                return self.X_name
            case _:
                raise ValueError(f'Bad encoding: {encoding}')

    def Y_offset(self, ref_index: str | int | None) -> Tensor:
        match ref_index:
            case None:
                return self.Y
            case 'centered':
                mask = torch.isfinite(self.Y)
                Y_count = mask.sum(dim=1)
                Y_sum = torch.where(mask, self.Y, 0).sum(dim=1)
                Y_mean = torch.where(Y_count > 0, Y_sum / Y_count, 0).unsqueeze(-1)
                return self.Y - Y_mean
            case _:
                if not isinstance(ref_index, int):
                    raise ValueError(f'Bad ref_index: {ref_index}')
                mask = torch.isfinite(self.Y[:, ref_index])
                new_Y = self.Y - self.Y[:, ref_index].unsqueeze(-1)
                new_Y[~mask, :] = float('nan')
                return new_Y

    def X_Von(self, X_encoding: Encoding, *, outlier_threshold: float | None = None,
              enough_states_frac: float = 0.7, range_V: tuple[float, float] = (-100, float('inf'))) -> Tensor:
        enough_Y_points = torch.isfinite(self.Y).sum(dim=1) >= self.encoding.n_outputs * enough_states_frac
        mask = torch.isfinite(self.Vons_V) & enough_Y_points & (self.Vons_V > range_V[0]) & (self.Vons_V < range_V[1])
        X = self.X_encoded(X_encoding)[mask]
        Y = self.Vons_V[mask]
        if outlier_threshold:
            z_mask = torch.abs((Y - torch.mean(Y)) / torch.std(Y)) < outlier_threshold
            X = X[z_mask]
            Y = Y[z_mask]
        return X, Y

    def X_Yi(self, i: int, X_encoding: Encoding, ref_index: int | str | None) -> tuple[tuple[str, ...], Tensor, Tensor]:
        '''Returns the X and Y for a specific task t'''
        Y = self.Y_offset(ref_index)
        mask = torch.isfinite(Y[:, i])
        Y = Y[mask, i]
        X = self.X_encoded(X_encoding)[mask]
        X_names = tuple(np.array(self.X_encoded(Encoding.NAME))[mask.tolist()])
        return X_names, X, Y

    def Xt_Y(self, X_encoding: Encoding, Y_encoding: Encoding, ref_index: int | str | None,
             ) -> tuple[tuple[str, ...], tuple[str, ...], Tensor, Tensor]:
        '''Returns the X and Y for all tasks, where the X includes a column for the task index'''
        X_names = []
        Y_names = []
        Xt = []
        Y = []
        for i in range(self.encoding.n_outputs):
            if i == ref_index:
                continue
            X_i_names, X_i, Y_i = self.X_Yi(i, X_encoding, ref_index)
            if Y_encoding == Encoding.INDEX:
                t_i = torch.full((X_i.shape[0], 1), i, dtype=X_i.dtype, device=X_i.device)
            elif Y_encoding == Encoding.PROP:
                t_i = torch.tensor(self.encoding.encodings[i]).unsqueeze(0).repeat(X_i.shape[0], 1)
            else:
                raise ValueError(f'Unsupported encoding: {Y_encoding}')
            X_names += X_i_names
            Y_names += len(X_i_names) * [self.encoding.adsorbates[i]]
            Xt.append(torch.cat((X_i, t_i), dim=1))
            Y.append(Y_i)
        Xt = torch.cat(Xt)
        Y = torch.cat(Y)
        return tuple(X_names), tuple(Y_names), Xt, Y

def Von(samples: Tensor, k_mask: Tensor, Ecycle_eV: float) -> Tensor:
    '''
    samples: Tensor of shape (N, M, K) where
        N is the number of samples from the posterior per material
        M is the number of materials
        K is the number of adsorbates
    k_mask: Tensor of shape (K, S) where
        K is the number of adsorbates
        S is the number of steps
    '''
    N, M, K = samples.shape
    _, S = k_mask.shape

    expanded_samples = samples.unsqueeze(-1).expand(N, M, K, S)
    expanded_k_mask = k_mask.unsqueeze(0).unsqueeze(0).expand(N, M, K, S)

    Ecurrent_eV, min_indices = torch.where(expanded_k_mask, expanded_samples, float('inf')).min(dim=2)
    Enext_eV = torch.roll(Ecurrent_eV, shifts=-1, dims=2)

    dE = Ecurrent_eV - Enext_eV
    dE[..., -1] -= Ecycle_eV

    turn_offs_V = dE.min(dim=2)[0]
    return turn_offs_V, min_indices


def Von_single(samples: Tensor, k_mask: dict[int, Tensor], Ecycle_eV: float) -> Tensor:
    turn_offs_V, min_indices = Von(samples.unsqueeze(0), k_mask, Ecycle_eV)
    return turn_offs_V.squeeze(0), min_indices.squeeze(0)
