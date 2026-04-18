from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler


class HydroSampler(BaseSampler):
    """Hydrological data sampler.

    Parameters
    ----------
    config
        Configuration dictionary.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.rho = config['model']['rho']
        self.warm_up = config['model'].get('warm_up', 0)

    def load_data(self):
        """Custom implementation for loading data."""
        print("Loading data...")

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        print("Preprocessing data...")

    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
    ) -> torch.Tensor:
        """Select a subset of input tensor."""
        batch_size, nx = len(i_grid), x.shape[-1]

        # Handle time indexing and create an empty tensor for selection
        if i_t is not None:
            x_tensor = torch.zeros(
                [self.rho + self.warm_up, batch_size, nx],
                device=self.device,
                requires_grad=has_grad,
            )
            for k in range(batch_size):
                x_tensor[:, k : k + 1, :] = x[
                    i_t[k] - self.warm_up : i_t[k] + self.rho,
                    i_grid[k] : i_grid[k] + 1,
                    :,
                ]
        else:
            x_tensor = (
                x[:, i_grid, :].float().to(self.device)
                if x.ndim == 3
                else x[i_grid, :].float().to(self.device)
            )

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            c_tensor = (
                c_tensor[i_grid].unsqueeze(1).repeat(1, self.rho + self.warm_up, 1)
            )
            return (
                (x_tensor, c_tensor)
                if tuple_out
                else torch.cat((x_tensor, c_tensor), dim=2)
            )

        return x_tensor

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch."""
        batch_size = self.config['train']['batch_size']
        i_sample, i_t = random_index(
            ngrid_train,
            nt,
            (batch_size, self.rho),
            warm_up=self.warm_up,
        )

        _core_keys = {'x_phy', 'c_phy', 'c_nn', 'xc_nn_norm', 'target'}

        sample = {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'c_phy': dataset['c_phy'][i_sample],
            'c_nn': dataset['c_nn'][i_sample],
            'xc_nn_norm': self.select_subset(
                dataset['xc_nn_norm'],
                i_sample,
                i_t,
                has_grad=False,
            ),
            'target': self.select_subset(dataset['target'], i_sample, i_t),
            'batch_sample': i_sample,
        }

        # Pass through extra keys (e.g. ac_all, elev_all, c_nn_norm).
        for key, value in dataset.items():
            if key in _core_keys or not isinstance(value, torch.Tensor):
                continue
            if value.ndim <= 2:
                sample[key] = value[i_sample]

        return sample

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        result = {}
        for key, value in dataset.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 3:
                result[key] = value[:, i_s:i_e, :].to(
                    dtype=torch.float32, device=self.device,
                )
            elif value.ndim == 2:
                result[key] = value[i_s:i_e, :].to(
                    dtype=torch.float32, device=self.device,
                )
            elif value.ndim == 1:
                result[key] = value[i_s:i_e].to(
                    dtype=torch.float32, device=self.device,
                )
        return result
