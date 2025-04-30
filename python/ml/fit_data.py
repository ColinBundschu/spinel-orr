from dataclasses import dataclass
import torch
from torch import Tensor
from scipy.stats import pearsonr
from botorch.models import SingleTaskGP

from ml.kernels import DFTKernel
from ml.scaler import Scaler


@dataclass(frozen=True)
class FitData:
    X_train: Tensor
    Y_train: Tensor
    Yhat_train_mean: Tensor
    Yhat_train_std: Tensor
    X_test: Tensor
    Y_test: Tensor
    Yhat_test_mean: Tensor
    Yhat_test_std: Tensor

    @property
    def MAE_train(self) -> float:
        """Mean Absolute Error for training data."""
        return (self.Y_train - self.Yhat_train_mean).abs().mean().item()

    @property
    def MAE_test(self) -> float | None:
        """Mean Absolute Error for test data."""
        if self.Yhat_test_mean is None or self.Y_test is None:
            return None
        return (self.Y_test - self.Yhat_test_mean).abs().mean().item()

    @property
    def R_train(self) -> float:
        """Pearson correlation coefficient for training data."""
        return pearsonr(self.Y_train.cpu().numpy().ravel(), self.Yhat_train_mean.cpu().numpy().ravel())[0]

    @property
    def R_test(self) -> float | None:
        """Pearson correlation coefficient for test data."""
        if self.Y_test.shape[0] < 2:
            return None
        return pearsonr(self.Y_test.cpu().numpy().ravel(), self.Yhat_test_mean.cpu().numpy().ravel())[0]


@dataclass(frozen=True)
class GPRFitData(FitData):
    N_features: int
    N_params: int
    _kernel: tuple
    norm_power_lambdas: tuple[tuple[float, float], ...]

    @classmethod
    def create(
        cls: 'GPRFitData',
        model: SingleTaskGP,
        X_train: Tensor,
        Y_train: Tensor,
        scaler_X: Scaler,
        scaler_Y: Scaler,
        *,
        X_test: Tensor | tuple[tuple[float, ...], ...] | None = None,
        Y_test: Tensor | tuple[float, ...] | None = None,
        norm_power_lambdas: tuple[tuple[float, float], ...] = tuple(),
    ) -> 'GPRFitData':

        if X_test is not None and X_test.numel() == 0:
            X_test = None
        if Y_test is not None and Y_test.numel() == 0:
            Y_test = None
        if (X_test is None) != (Y_test is None):
            raise ValueError("X_test and Y_test must be provided together.")

        with torch.no_grad():
            posterior = model.posterior(model.train_inputs[0])
            Yhat_mean = scaler_Y.inverse_transform(posterior.mean)

            if X_test is None:
                Yhat_test_mean = None
            else:
                posterior_test = model.posterior(scaler_X.transform(X_test))
                Yhat_test_mean = scaler_Y.inverse_transform(posterior_test.mean)

        kernel_tuple = tuple(item for item in model.covar_module.serialize().items())

        return cls(
            X_train,
            Y_train,
            Yhat_mean,
            None,
            X_test,
            Y_test,
            Yhat_test_mean,
            None,
            model.covar_module.N,
            model.covar_module.N_sigmas,
            kernel_tuple,
            norm_power_lambdas,
        )

    @property
    def kernel(self) -> DFTKernel:
        """Reconstruct the kernel from serialized data."""
        return DFTKernel(**dict(self._kernel))

@dataclass
class Metric():
    fit_full: FitData
    Yhat_test_mean: Tensor
    Y_test: Tensor
    MAE_test: float
    R_test: float
