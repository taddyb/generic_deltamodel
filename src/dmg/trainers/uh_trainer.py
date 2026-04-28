"""Trainer for dHBV2.0UH multiscale training.

Implements the gauge-centric training loop where:
  1. Data is loaded as distributed chunks via MtsHydroLoader.
  2. UhHydroSampler yields (time_window, gauge_batch) pairs.
  3. UhModelHandler runs NN → Hbv_2 → accumulate → gauge-scale loss.
  4. Gradients propagate from gauge loss through accumulation to all
     upstream sub-basin parameters.

Reference:
    Song, Y., Bindas, T., Shen, C., et al. (2025).
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import tqdm

from dmg.core.calc import Metrics
from dmg.core.data.samplers.uh_hydro_sampler import UhHydroSampler
from dmg.core.utils.utils import save_model, save_train_state
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class UhTrainer(BaseTrainer):
    """Trainer for dHBV2.0UH multiscale training with gauge-scale loss.

    Parameters
    ----------
    config
        Configuration settings.
    model
        UhModelHandler instance.
    data_loader
        MtsHydroLoader instance (provides distributed data with topology).
    verbose
        Whether to print verbose output.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        data_loader: Any = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.verbose = verbose
        self.device = config['device']
        self.sampler = UhHydroSampler(config)

        self.epochs = config['train']['epochs']
        self.start_epoch = config['train']['start_epoch'] + 1
        self.save_epoch = config['train']['save_epoch']
        self.warm_up = config['model'].get('warm_up', 365)

        self.optimizer = None
        self.scheduler = None
        self.loss_func = None

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer from config."""
        name = self.config['train']['optimizer']['name']
        lr = self.config['train']['lr']
        optimizer_map = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
        }
        cls = optimizer_map.get(name)
        if cls is None:
            raise ValueError(f"Unknown optimizer: {name}")
        self.optimizer = cls(self.model.get_parameters(), lr=lr)
        return self.optimizer

    def init_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Initialize LR scheduler from config."""
        params = self.config['train'].get('lr_scheduler', {})
        if not params or params.get('name', 'none') == 'none':
            return None

        params = params.copy()
        name = params.pop('name')
        scheduler_map = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        cls = scheduler_map.get(name)
        if cls is None:
            raise ValueError(f"Unknown scheduler: {name}")
        self.scheduler = cls(self.optimizer, **params)
        return self.scheduler

    def init_loss_func(self, target_obs: torch.Tensor) -> None:
        """Initialize loss function with gauge-scale observation stats.

        Parameters
        ----------
        target_obs
            Gauge-scale observations [n_gauges, T] for computing per-gauge
            standard deviations used in NseBatchLoss normalization.
        """
        from dmg.core.utils.factory import load_criterion

        # NseBatchLoss expects y_obs as [T, n_gauges, 1]
        y_obs = target_obs.T.unsqueeze(-1)
        self.loss_func = load_criterion(
            y_obs,
            self.config['train']['loss_function'],
            device=self.device,
        )
        self.model.loss_func = self.loss_func

    def train(self) -> None:
        """Main training loop (matches lumped Trainer pattern).

        Each epoch runs n_minibatch random (gauge, time) samples, where
        n_minibatch is computed for ~99% data coverage — same formula
        as the lumped model's `create_training_grid`.
        """
        log.info(
            f"dHBV2.0UH Training: {self.start_epoch} to {self.epochs} epochs",
        )

        # Load training data
        self.data_loader.load_dataset(mode='train')
        self.train_data = None
        for chunk in self.data_loader.get_dataset():
            self.train_data = chunk
            break
        if self.train_data is None:
            raise ValueError("No training data available.")

        # Compute minibatches per epoch (same formula as lumped)
        n_gauges = len(self.train_data.gauge)
        n_t = self.train_data.dyn_input.shape[1]
        self.n_minibatch = self.sampler.compute_n_minibatch(n_gauges, n_t)
        log.info(
            f"Training grid: {n_gauges} gauges, {n_t} timesteps, "
            f"{self.n_minibatch} minibatches/epoch",
        )

        # Initialize loss function from gauge-scale observations
        self.init_loss_func(self.train_data.target)

        # Initialize optimizer and scheduler
        if self.optimizer is None:
            self.init_optimizer()
        self.init_scheduler()

        if self.start_epoch > 1:
            self._load_checkpoint()

        csv_log = os.path.join(self.config.get('output_dir', '.'), 'training_log.csv')
        os.makedirs(os.path.dirname(csv_log), exist_ok=True)
        with open(csv_log, 'w') as f:
            f.write('epoch,batch,loss,time_s,gpu_mem_mb\n')

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss = self._train_one_epoch(epoch, csv_log)

            if self.scheduler is not None:
                self.scheduler.step()

            log.info(f"Epoch {epoch}/{self.epochs} | loss={epoch_loss:.6f}")

            if epoch % self.save_epoch == 0:
                self.model.save_model(epoch)
                save_train_state(
                    self.config['model_dir'],
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    clear_prior=True,
                )

        log.info("Training complete.")

    def _train_one_epoch(self, epoch: int, csv_log: str) -> float:
        """Run one training epoch with random (gauge, time) sampling.

        Matches the lumped Trainer pattern: n_minibatch random samples
        per epoch, each picking batch_size gauges and a random time window.

        Parameters
        ----------
        epoch
            Current epoch number.
        csv_log
            Path to CSV log file.

        Returns
        -------
        float
            Mean batch loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        if hasattr(self.model, 'loss_dict'):
            for key in self.model.loss_dict:
                self.model.loss_dict[key] = 0.0

        prog_bar = tqdm.tqdm(
            range(1, self.n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for mb in prog_bar:
            batch_start = time.perf_counter()

            # Random sample: batch_size gauges + random time window
            batch = self.sampler.get_training_sample(self.train_data)

            # Move to device
            batch_gpu = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward + loss + backward
            self.optimizer.zero_grad()
            self.model(batch_gpu)
            loss = self.model.calc_loss(batch_gpu)

            batch_loss = loss.item()

            # Skip NaN batches — don't corrupt optimizer state
            if not np.isfinite(batch_loss):
                log.warning(f"  NaN/inf loss at batch {mb}, skipping")
                self.optimizer.zero_grad()
                del batch_gpu, loss
                if self.device != 'cpu':
                    torch.cuda.empty_cache()
                continue

            loss.backward()
            clip_norm = self.config['train'].get('clip_gradient_norm')
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_parameters(), max_norm=clip_norm,
                )
            self.optimizer.step()

            total_loss += batch_loss

            # Free GPU memory between batches
            del batch_gpu, loss
            if self.device != 'cpu':
                torch.cuda.empty_cache()

            # Log
            elapsed = time.perf_counter() - batch_start
            mem = 0
            if self.device != 'cpu':
                mem = int(torch.cuda.memory_reserved(device=self.device) * 1e-6)
            with open(csv_log, 'a') as f:
                f.write(f"{epoch},{mb},{batch_loss:.6f},{elapsed:.2f},{mem}\n")

            prog_bar.set_postfix(loss=f"{batch_loss:.4f}")

        return total_loss / self.n_minibatch

    def evaluate(self) -> pd.DataFrame:
        """Evaluate model on validation/test data.

        Processes gauges in spatial batches and time in chunks of
        rho+warm_up to stay within GPU memory. Predictions are
        concatenated across time chunks for full-period metrics.

        Returns gauge-level metrics (NSE, KGE, etc.).
        """
        self.model.eval()
        pred_dict = {}
        obs_dict = {}

        rho = self.config['model']['rho']
        warm_up = self.warm_up

        self.data_loader.load_dataset(mode='test')
        for chunk_data in self.data_loader.get_dataset():
            n_gauges = len(chunk_data.gauge)
            n_t = chunk_data.dyn_input.shape[1]
            eval_batch_size = self.config.get('test', {}).get('batch_size', 4)

            with torch.no_grad():
                for g_start in range(0, n_gauges, eval_batch_size):
                    g_end = min(g_start + eval_batch_size, n_gauges)
                    gauge_idx = list(range(g_start, g_end))

                    # Process in time chunks to avoid OOM
                    all_pred = []
                    all_obs = []

                    # First chunk starts at warm_up (need warm_up days before)
                    for t_start in range(warm_up, n_t, rho):
                        t_end = min(t_start + rho, n_t)
                        t0 = t_start - warm_up  # include warmup

                        batch = self.sampler._build_batch(
                            chunk_data, t_start, gauge_idx,
                        )
                        batch_gpu = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        self.model(batch_gpu, eval=True)

                        q_gauge = self.model.q_gauge.cpu()  # [warm_up+chunk, n_gauge]
                        target = batch['target']  # [n_gauge, warm_up+chunk]

                        # Take only the prediction part (skip warmup)
                        all_pred.append(q_gauge[warm_up:, :].numpy())
                        all_obs.append(target[:, warm_up:].numpy())

                        del batch_gpu
                        if self.device != 'cpu':
                            torch.cuda.empty_cache()

                    # Concatenate time chunks
                    if all_pred:
                        full_pred = np.concatenate(all_pred, axis=0)  # [T_total, n_gauge]
                        full_obs = np.concatenate(all_obs, axis=1)  # [n_gauge, T_total]

                        gauge_indices = list(range(g_start, g_end))
                        global_indices = chunk_data.gauge_index[gauge_indices]
                        for i, gidx in enumerate(global_indices):
                            gidx = int(gidx)
                            pred_dict[gidx] = full_pred[:, i]
                            obs_dict[gidx] = full_obs[i, :]

            log.info(f"Evaluated {n_gauges} gauges in time chunks of {rho} days")

        # Compute per-gauge metrics
        metrics_list = []
        for gidx in sorted(pred_dict.keys()):
            pred = pred_dict[gidx]
            obs = obs_dict[gidx]
            m = Metrics(pred=pred, target=obs).model_dump()
            m.pop('pred', None)
            m.pop('target', None)
            m = {k: v[0] if isinstance(v, (list, np.ndarray)) else v for k, v in m.items()}
            m['gauge_idx'] = gidx
            metrics_list.append(m)

        df = pd.DataFrame(metrics_list)
        if len(df) > 0:
            log.info(
                f"Eval: {len(df)} gauges | "
                f"median NSE={df.get('nse', pd.Series([0])).median():.3f} | "
                f"median KGE={df.get('kge', pd.Series([0])).median():.3f}",
            )
        return df

    def _load_checkpoint(self) -> None:
        """Load optimizer/scheduler state from checkpoint."""
        model_dir = self.config['model_dir']
        target_epoch = self.start_epoch - 1
        for file in os.listdir(model_dir):
            if file == f"trainer_state_ep{target_epoch}.pt":
                path = os.path.join(model_dir, file)
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                log.info(f"Loaded trainer state from epoch {target_epoch}")
                return
        log.warning(f"No checkpoint found for epoch {target_epoch}")

    def inference(self) -> dict[str, np.ndarray]:
        """Run inference on simulation data. Not used for UH training."""
        raise NotImplementedError(
            "Use MsTrainer.inference() for sub-basin-scale simulation.",
        )

    def calc_metrics(self) -> None:
        """Compute and save metrics."""
        df = self.evaluate()
        out_dir = self.config.get('output_dir', '.')
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, 'gauge_metrics.csv'), index=False)
