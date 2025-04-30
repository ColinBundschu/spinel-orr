from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class Scaler(ABC):
    """Abstract base class for data scalers."""

    @classmethod
    @abstractmethod
    def fit(cls, data: Tensor) -> 'Scaler':
        """Fit the scaler to the data and return an instance of the scaler."""

    @abstractmethod
    def transform(self, data: Tensor) -> Tensor:
        """Transform the data using the fitted scaler."""

    @abstractmethod
    def inverse_transform(self, data: Tensor) -> Tensor:
        """Inverse transform the scaled data."""


@dataclass(frozen=True)
class ZeroToOneScaler(Scaler):
    min: Tensor
    max: Tensor

    @classmethod
    def fit(cls, data: Tensor) -> 'ZeroToOneScaler':
        return cls(data.min(dim=0, keepdim=True).values, data.max(dim=0, keepdim=True).values)

    def transform(self, data: Tensor) -> Tensor:
        data_range = self.max - self.min
        if torch.any(data_range == 0):
            raise ValueError("Cannot scale data with zero range. Ensure `max` and `min` are distinct.")
        return (data - self.min) / data_range

    def inverse_transform(self, data: Tensor) -> Tensor:
        return data * (self.max - self.min) + self.min


@dataclass(frozen=True)
class Mean0Var1Scaler(Scaler):
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, data: Tensor) -> 'Mean0Var1Scaler':
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True, unbiased=False)
        if torch.any(std == 0):
            raise ValueError("Cannot scale data with zero variance. Ensure the data has non-zero variance.")
        return cls(mean, std)

    def transform(self, data: Tensor) -> Tensor:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: Tensor) -> Tensor:
        return data * self.std + self.mean


@dataclass(frozen=True)
class NegLogScaler(Scaler):
    scaler: Mean0Var1Scaler

    @classmethod
    def fit(cls, data: Tensor) -> 'NegLogScaler':
        max_value = data.max()
        if max_value >= 0:
            raise ValueError(f'Data must be negative for log scaling, but max value is {max_value}.')
        scaler = Mean0Var1Scaler.fit(torch.log(-data))
        return cls(scaler)

    def transform(self, data: Tensor) -> Tensor:
        return self.scaler.transform(torch.log(-data))

    def inverse_transform(self, data: Tensor) -> Tensor:
        return -torch.exp(self.scaler.inverse_transform(data))

@dataclass(frozen=True)
class PassthroughScaler(Scaler):
    @classmethod
    def fit(cls, data: Tensor) -> 'PassthroughScaler':
        return cls()

    def transform(self, data: Tensor) -> Tensor:
        return data

    def inverse_transform(self, data: Tensor) -> Tensor:
        return data
