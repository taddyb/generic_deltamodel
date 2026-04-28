"""Model handler for dHBV2.0UH multiscale training.

Extends ModelHandler to add network flow accumulation after the standard
DplModel (NN → Hbv_2) forward pass. Per-basin runoff is accumulated to
gauge outlets via a reachability matrix multiply, then loss is computed
at gauge scale.

The accumulation step is differentiable (a simple matmul), so gradients
from gauge-scale loss propagate back through Hbv_2 and the NN to every
upstream sub-basin's parameters.

Reference:
    Song, Y., Bindas, T., Shen, C., et al. (2025).
    High-resolution national-scale water modeling is enhanced by multiscale
    differentiable physics-informed machine learning.
    Water Resources Research, 61, e2024WR038928.
"""

import logging
from typing import Any, Optional

import torch

from dmg.models.model_handler import ModelHandler

log = logging.getLogger(__name__)


class UhModelHandler(ModelHandler):
    """Extends ModelHandler to accumulate per-basin runoff to gauges.

    After the standard DplModel forward pass produces Q_local (mm/day) at
    each sub-basin, this handler:
      1. Converts mm/day → m³/s using local catchment area.
      2. Accumulates to gauge outlets via:  Q_gauge = topo @ Q_local_m3s
         where topo is the [n_gauge, n_units] reachability matrix.
      3. Computes loss at gauge scale using the accumulated Q_gauge.

    The within-basin gamma UH (Hbv_2, routing=True) implicitly learns
    travel time: the NN sets route_a/route_b as functions of basin
    attributes, so distant basins get longer, more spread-out UH shapes.
    """

    MM_D_TO_M3S = 1000.0 / 86400.0  # mm/day per km² → m³/s

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(config, device=device, verbose=verbose)
        self.warm_up = config['model'].get('warm_up', 0)
        self.gpu_sub_batch = config.get('gpu_sub_batch', 100)
        self.q_gauge = None

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with GPU sub-batching and network accumulation.

        Processes sub-basins in chunks of `gpu_sub_batch` through the
        NN + Hbv_2, then accumulates to gauge scale.

        Parameters
        ----------
        dataset_dict
            Must include 'areas' [n_units] and 'topo' [n_gauge, n_units]
            in addition to the standard DplModel inputs.
        eval
            Whether to run in evaluation mode (no gradients).

        Returns
        -------
        dict[str, torch.Tensor]
            Standard DplModel output dict (per-basin fluxes/states).
            The accumulated Q_gauge is stored in self.q_gauge for loss.
        """
        n_units = dataset_dict['xc_nn_norm'].shape[1]
        T = dataset_dict['xc_nn_norm'].shape[0]
        sb = self.gpu_sub_batch

        if n_units <= sb:
            # Small enough to run in one pass
            output_dict = super().forward(dataset_dict, eval=eval)
        else:
            # Sub-batch through NN + Hbv_2 to fit in GPU memory
            model_name = self.models[0]
            dpl_model = self.model_dict[model_name]

            if eval:
                dpl_model.eval()
            else:
                dpl_model.train()

            all_streamflow = []
            for i in range(0, n_units, sb):
                j = min(i + sb, n_units)
                sub_dict = {
                    'xc_nn_norm': dataset_dict['xc_nn_norm'][:, i:j, :],
                    'c_nn_norm': dataset_dict['c_nn_norm'][i:j, :],
                    'x_phy': dataset_dict['x_phy'][:, i:j, :],
                    'ac_all': dataset_dict['ac_all'][i:j],
                    'elev_all': dataset_dict['elev_all'][i:j],
                }

                if eval:
                    with torch.no_grad():
                        sub_out = dpl_model(sub_dict)
                else:
                    sub_out = dpl_model(sub_dict)

                all_streamflow.append(sub_out['streamflow'])

            # Concatenate sub-batch streamflows [T, n_units, 1]
            merged_streamflow = torch.cat(all_streamflow, dim=1)
            output_dict = {model_name: {'streamflow': merged_streamflow}}
            self.output_dict = output_dict

        # Extract per-basin streamflow [T, n_units, 1]
        model_name = list(output_dict.keys())[0]
        q_local = output_dict[model_name]['streamflow']

        # Convert mm/day → m³/s using local catchment area
        areas = dataset_dict['areas']
        q_m3s = q_local[:, :, 0] * areas.unsqueeze(0) * self.MM_D_TO_M3S

        # Accumulate to gauge outlets via reachability matrix
        topo = dataset_dict['topo']
        self.q_gauge = torch.matmul(topo, q_m3s.T).T  # [T, n_gauge]

        return output_dict

    def calc_loss(
        self,
        dataset_dict: dict[str, torch.Tensor],
        loss_func: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Calculate loss at gauge scale using accumulated Q_gauge.

        Parameters
        ----------
        dataset_dict
            Must include 'target' [n_gauge, T] and 'batch_sample'.
        loss_func
            Loss function to use. Falls back to self.loss_func.

        Returns
        -------
        torch.Tensor
            Gauge-scale loss.
        """
        if self.q_gauge is None:
            raise RuntimeError(
                "forward() must be called before calc_loss(). "
                "No accumulated Q_gauge available.",
            )

        if not self.loss_func and not loss_func:
            raise ValueError("No loss function defined.")
        loss_func = loss_func or self.loss_func

        # q_gauge: [T, n_gauge], target: [n_gauge, T] → [T, n_gauge]
        # Target has NaN during warmup, which NseBatchLoss masks out.
        prediction = self.q_gauge
        target = dataset_dict['target'].T

        loss = loss_func(
            prediction,
            target,
            sample_ids=dataset_dict['batch_sample'],
        )

        # Track per-model loss for logging
        model_name = list(self.output_dict.keys())[0]
        self.loss_dict[model_name] += loss.item()

        return loss
