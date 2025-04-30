from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class AdsorbatesEncoding():
    name: str
    n_steps: int
    encodings: tuple[tuple[int, ...], ...]
    step_indices: tuple[int, ...]
    adsorbates: tuple[str, ...]

    def __post_init__(self):
        if not len(self.encodings) == len(self.adsorbates) == len(self.step_indices):
            raise ValueError('The number of encodings, adsorbates and step indices must be the same.')

        if not all(len(encoding) == len(self.encodings[0]) for encoding in self.encodings):
            raise ValueError('All encodings must have the same length.')

    @property
    def n_outputs(self):
        return len(self.step_indices)

    @property
    def n_features(self):
        return len(self.encodings[0])

    @property
    def step_mask(self) -> torch.Tensor:
        step_mask_tensor = torch.zeros(self.n_outputs, self.n_steps, dtype=torch.bool)
        for t, step_index in enumerate(self.step_indices):
            step_mask_tensor[t, step_index] = True
        return step_mask_tensor
