"""Train dHBV2.0UH: multiscale differentiable HBV with gauge-scale loss.

Runs the Song et al. (2025) multiscale training paradigm:
  1. MtsHydroLoader loads distributed hourly forcings, attributes, topology,
     and gauge observations in yearly chunks.
  2. UhHydroSampler yields gauge-centric batches (gauge subset + upstream
     sub-basin union + reachability matrix).
  3. UhModelHandler runs LstmMlpModel → Hbv_2 → flow accumulation via
     reachability matrix multiply → gauge-scale discharge.
  4. NseBatchLoss at gauge scale, gradients propagate back through
     accumulation to all upstream sub-basin parameters.

Usage:
    python scripts/train_dhbv2_uh.py \\
        --config conf/config_dhbv_2_uh.yaml \\
        --output-dir output/dhbv2_uh_run1

Requirements:
    - Topology JSON: {nodes, edges, gage_hf} for MERIT network
    - Hourly forcing NetCDFs: one per year in path_forcing/
    - Basin attributes NetCDF
    - USGS daily streamflow NetCDF
    - Gauge info CSV (dhbv2_gages.csv)

Reference:
    Song, Y., Bindas, T., Shen, C., et al. (2025).
    High-resolution national-scale water modeling is enhanced by multiscale
    differentiable physics-informed machine learning.
    Water Resources Research, 61, e2024WR038928.
"""

import argparse
import logging
import os
import sys
import time

import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train dHBV2.0UH with multiscale gauge-scale loss',
    )
    parser.add_argument(
        '--config', required=True, type=str,
        help='Path to config YAML (e.g., conf/config_dhbv_2_uh.yaml)',
    )
    parser.add_argument(
        '--output-dir', default=None, type=str,
        help='Output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--device', default=None, type=str,
        help='Device override (e.g., cuda:0, cpu)',
    )
    parser.add_argument(
        '--epochs', default=None, type=int,
        help='Override number of training epochs',
    )
    parser.add_argument(
        '--batch-size', default=None, type=int,
        help='Override gauge batch size',
    )
    parser.add_argument(
        '--resume-epoch', default=None, type=int,
        help='Resume training from this epoch',
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Skip training, run evaluation on test period only',
    )
    parser.add_argument(
        '--eval-epoch', default=None, type=int,
        help='Epoch to load for evaluation (default: test.test_epoch from config)',
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Acceptance test: 2 epochs, 2 gauges, 1-year window, first 50 sub-basins only',
    )
    args = parser.parse_args()

    # ── Load and resolve config ──────────────────────────────────────
    raw_config = OmegaConf.load(args.config)

    # Hydra defaults aren't resolved by OmegaConf.load(), so manually
    # merge the observation config referenced in 'defaults'.
    config_dir = os.path.dirname(os.path.abspath(args.config))
    defaults = OmegaConf.to_container(raw_config.get('defaults', []))
    for entry in defaults:
        if isinstance(entry, dict):
            for key, val in entry.items():
                if key in ('_self_', 'hydra'):
                    continue
                obs_path = os.path.join(config_dir, key, f'{val}.yaml')
                if os.path.isfile(obs_path):
                    obs_cfg = OmegaConf.load(obs_path)
                    raw_config = OmegaConf.merge(raw_config, {key: obs_cfg})

    config = OmegaConf.to_container(raw_config, resolve=True)
    config.pop('defaults', None)

    # Quick acceptance test: minimal config to flush out bugs fast
    if args.quick_test:
        log.info("*** QUICK TEST MODE: 2 epochs, 10 gauges, short windows ***")
        config['train']['epochs'] = 2
        config['train']['batch_size'] = 2
        config['train']['save_epoch'] = 1
        config['train']['start_time'] = '1981/10/01'
        config['train']['end_time'] = '1982/09/30'
        config['test']['start_time'] = '1982/10/01'
        config['test']['end_time'] = '1983/09/30'
        config['model']['rho'] = 30       # 30-day prediction window
        config['model']['warm_up'] = 30   # 30-day warmup (minimal)
        config.setdefault('_quick_test', True)
        config['_quick_test_max_gauges'] = 10

    # Apply CLI overrides
    if args.output_dir:
        config['model_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['output_dir'] = args.output_dir
    else:
        run_name = f"dhbv2_uh_{time.strftime('%Y%m%d_%H%M%S')}"
        config.setdefault('output_dir', f'output/{run_name}')
        config.setdefault('model_dir', f'output/{run_name}/checkpoints')

    if args.device:
        config['device'] = args.device
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.resume_epoch is not None:
        config['train']['start_epoch'] = args.resume_epoch

    # Ensure output dirs exist
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    # Set device
    if config['device'] == 'cuda' and torch.cuda.is_available():
        gpu_id = config.get('gpu_id', 0)
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        config['device'] = device
    elif config['device'] != 'cpu':
        config['device'] = 'cpu'
        log.warning("CUDA not available, falling back to CPU")

    config['mode'] = 'train'
    config.setdefault('multimodel_type', None)
    config.setdefault('cache_states', False)

    # Propagate cache_states into sub-model configs
    cache_states = config.get('cache_states', False)
    if config.get('model', {}).get('nn'):
        config['model']['nn'].setdefault('cache_states', cache_states)
    if config.get('model', {}).get('phy'):
        config['model']['phy'].setdefault('cache_states', cache_states)
        config['model']['phy'].setdefault(
            'warm_up', config['model'].get('warm_up', 365),
        )

    torch.set_float32_matmul_precision('high')

    log.info("=" * 60)
    log.info("dHBV2.0UH Multiscale Training")
    log.info("=" * 60)
    log.info(f"Device: {config['device']}")
    log.info(f"Epochs: {config['train']['epochs']}")
    log.info(f"Gauge batch size: {config['train']['batch_size']}")
    log.info(f"Output: {config['output_dir']}")
    log.info(f"Model dir: {config['model_dir']}")

    # ── Initialize data loader ───────────────────────────────────────
    from dmg.core.data.loaders.uh_hydro_loader import UhHydroLoader

    log.info("Loading distributed data (topology, forcings, observations)...")
    data_loader = UhHydroLoader(config)

    # ── Initialize model ─────────────────────────────────────────────
    from dmg.models.uh_model_handler import UhModelHandler

    # For eval-only, load the specified epoch checkpoint
    if args.eval_only:
        eval_epoch = args.eval_epoch or config['test']['test_epoch']
        config['mode'] = 'test'
        config['test']['test_epoch'] = eval_epoch
        log.info(f"Eval-only mode: loading epoch {eval_epoch} from {config['model_dir']}")

    log.info("Initializing UhModelHandler (LstmMlpModel + Hbv_2 + accumulation)...")
    model = UhModelHandler(config, verbose=True)
    model.to(config['device'])

    n_params = sum(p.numel() for p in model.get_parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    if 'cuda' in config['device']:
        mem = torch.cuda.memory_reserved(device=config['device']) * 1e-6
        log.info(f"GPU memory after model init: {mem:.0f} MB")

    # ── Initialize trainer ───────────────────────────────────────────
    from dmg.trainers.uh_trainer import UhTrainer

    trainer = UhTrainer(
        config=config,
        model=model,
        data_loader=data_loader,
        verbose=True,
    )

    # ── Train or Eval ────────────────────────────────────────────────
    if args.eval_only:
        log.info("Running evaluation...")
        metrics = trainer.evaluate()
        metrics_path = os.path.join(config['output_dir'], 'gauge_metrics.csv')
        metrics.to_csv(metrics_path, index=False)
        log.info(f"Metrics saved to {metrics_path}")
        if len(metrics) > 0:
            for col in ['nse', 'kge', 'bias', 'flv', 'fhv']:
                if col in metrics.columns:
                    log.info(f"  {col}: median={metrics[col].median():.3f}")
    else:
        start = time.perf_counter()
        trainer.train()
        elapsed = (time.perf_counter() - start) / 60
        log.info(f"Training completed in {elapsed:.1f} minutes")

    # ── Evaluate ─────────────────────────────────────────────────────
    if 'test' in config.get('mode', 'train'):
        log.info("Running evaluation...")
        metrics = trainer.evaluate()
        metrics_path = os.path.join(config['output_dir'], 'gauge_metrics.csv')
        metrics.to_csv(metrics_path, index=False)
        log.info(f"Metrics saved to {metrics_path}")

    # ── Cleanup ──────────────────────────────────────────────────────
    if 'cuda' in config['device']:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == '__main__':
    main()
