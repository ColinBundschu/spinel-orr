import itertools
import math

import torch
from gpytorch.kernels import Kernel
from torch import Tensor
from torch.nn import Parameter, ParameterList


class DFTKernel(Kernel):
    """
    Custom kernel implementation for the covariance function k(x1, x2) 
    for DFT Free energy calculations.
    """

    def __init__(self, N: int, depth: int, **kwargs):
        super().__init__()
        self.N = N
        self.depth = depth
        self.sigmas = ParameterList()
        self.N_sigmas = 0  # Populated in the for loop below

        for d in range(self.depth + 1):
            kwargs_sigmas_d = kwargs.get(f'sigmas_{d}', None)
            if kwargs_sigmas_d is not None:
                # Check if the length of the provided sigma is correct
                if len(kwargs_sigmas_d) != math.comb(N, d):
                    raise ValueError(f"Invalid {d} length {kwargs_sigmas_d}, expected {math.comb(N, d)}")
                sigmas_d = torch.tensor(kwargs_sigmas_d, dtype=torch.double)
            else:
                sigmas_d = torch.ones(math.comb(N, d), dtype=torch.double)

            self.sigmas.append(ParameterList())
            for sigma in sigmas_d:
                if sigma == 0:
                    self.sigmas[d].append(Parameter(sigma, requires_grad=False))
                else:
                    self.sigmas[d].append(Parameter(sigma, requires_grad=True))
                    self.N_sigmas += 1

    def serialize(self) -> dict:
        """
        Serialize the DFTKernel into an dict of immutable representations.
        """
        data = {}
        data['N'] = self.N
        data['depth'] = self.depth
        for d, sigmas in enumerate(self.sigmas):
            data[f'sigmas_{d}'] = tuple([sigma.item() for sigma in sigmas])
        return data

    def freeze_zeros(self, cutoff: float = 1e-6) -> float:
        self.N_sigmas = 0
        min_val = float('inf')
        for d in range(self.depth + 1):
            for sigma in self.sigmas[d]:
                if abs(sigma.item()) < cutoff:
                    sigma.requires_grad = False
                    sigma.data = torch.zeros_like(sigma)
                else:
                    sigma.requires_grad = True
                    self.N_sigmas += 1
                    min_val = min(min_val, abs(sigma.item()))

        return min_val

    def L_norm(self, power: float) -> Tensor:
        """
        Computes the L_power norm of all parameters in the kernel.
        """
        if power == 0:
            return sum(param != 0 for param_list in self.sigmas for param in param_list)
        if power % 2 == 0:
            return sum(param.pow(power) for param_list in self.sigmas for param in param_list)
        return sum(param.abs().pow(power) for param_list in self.sigmas for param in param_list)

    def to(self, device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format):
        super().to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format)
        return self

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        covar_matrix = torch.zeros((n1, n2), dtype=x1.dtype, device=x1.device)
        for d in range(self.depth + 1):
            combinations = torch.tensor(list(itertools.combinations(range(self.N), d)), dtype=torch.int64)
            match_matrix_d = (x1[:, combinations].unsqueeze(1) == x2[:, combinations].unsqueeze(0)).all(dim=-1)
            covar_matrix_d = torch.stack([sigma.view(1, 1) for sigma in self.sigmas[d]], dim=-1) ** 2
            covar_matrix += torch.sum(match_matrix_d * covar_matrix_d, dim=-1)

        return covar_matrix.diag() if diag else covar_matrix
