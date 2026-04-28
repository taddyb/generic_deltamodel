"""Diagnostic script for dHBV2.0UH multiscale training.

Runs two experiments to verify the pipeline is correct:
  Exp 1: Train on 10 gauges, eval on SAME data → should give high NSE (overfit)
  Exp 2: Train on 10 gauges, eval on DIFFERENT period → should give reasonable NSE

Prints detailed diagnostics: prediction vs target magnitudes, accumulation
sanity checks, and per-gauge metrics.

Usage:
    uv run python scripts/diagnose_uh.py --config conf/config_dhbv_2_uh.yaml
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
)
log = logging.getLogger('diagnose')


def load_config(config_path):
    raw = OmegaConf.load(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    defaults = OmegaConf.to_container(raw.get('defaults', []))
    for entry in defaults:
        if isinstance(entry, dict):
            for key, val in entry.items():
                if key in ('_self_', 'hydra'):
                    continue
                obs_path = os.path.join(config_dir, key, f'{val}.yaml')
                if os.path.isfile(obs_path):
                    obs_cfg = OmegaConf.load(obs_path)
                    raw = OmegaConf.merge(raw, {key: obs_cfg})
    config = OmegaConf.to_container(raw, resolve=True)
    config.pop('defaults', None)
    return config


def setup_config(config, output_dir, train_start, train_end, test_start, test_end):
    config['train']['start_time'] = train_start
    config['train']['end_time'] = train_end
    config['test']['start_time'] = test_start
    config['test']['end_time'] = test_end
    config['train']['epochs'] = 30
    config['train']['batch_size'] = 2
    config['train']['save_epoch'] = 10
    config['model']['rho'] = 365
    config['model']['warm_up'] = 365
    config['model_dir'] = os.path.join(output_dir, 'checkpoints')
    config['output_dir'] = output_dir
    config['mode'] = 'train'
    config['_quick_test_max_gauges'] = 10
    config.setdefault('multimodel_type', None)
    config.setdefault('cache_states', False)
    config.setdefault('gpu_sub_batch', 100)

    if config.get('model', {}).get('nn'):
        config['model']['nn'].setdefault('cache_states', False)
    if config.get('model', {}).get('phy'):
        config['model']['phy'].setdefault('cache_states', False)
        config['model']['phy'].setdefault('warm_up', 365)

    if config['device'] == 'cuda' and torch.cuda.is_available():
        gpu_id = config.get('gpu_id', 0)
        config['device'] = f'cuda:{gpu_id}'
        torch.cuda.set_device(config['device'])
    else:
        config['device'] = 'cpu'

    return config


def diagnose_predictions(pred_dict, obs_dict, label):
    """Print diagnostic stats comparing predictions to observations."""
    log.info(f"\n{'='*60}")
    log.info(f"DIAGNOSTICS: {label}")
    log.info(f"{'='*60}")

    all_pred = []
    all_obs = []
    nse_list = []

    for gidx in sorted(pred_dict.keys()):
        pred = pred_dict[gidx]
        obs = obs_dict[gidx]

        # Remove NaN from obs
        valid = ~np.isnan(obs)
        pred_v = pred[valid]
        obs_v = obs[valid]

        if len(obs_v) < 10:
            log.info(f"  Gauge {gidx}: too few valid obs ({len(obs_v)})")
            continue

        # NSE
        ss_res = np.sum((pred_v - obs_v) ** 2)
        ss_tot = np.sum((obs_v - np.mean(obs_v)) ** 2)
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        ratio = np.median(pred_v) / np.median(obs_v) if np.median(obs_v) != 0 else np.inf

        log.info(
            f"  Gauge {gidx}: NSE={nse:7.3f} | "
            f"pred median={np.median(pred_v):8.2f} obs median={np.median(obs_v):8.2f} | "
            f"pred/obs ratio={ratio:.3f} | "
            f"pred range=[{pred_v.min():.1f}, {pred_v.max():.1f}] "
            f"obs range=[{obs_v.min():.1f}, {obs_v.max():.1f}]"
        )

        all_pred.extend(pred_v)
        all_obs.extend(obs_v)
        nse_list.append(nse)

    if nse_list:
        log.info(f"\n  SUMMARY ({len(nse_list)} gauges):")
        log.info(f"    Median NSE: {np.median(nse_list):.3f}")
        log.info(f"    Mean NSE:   {np.mean(nse_list):.3f}")
        log.info(f"    Overall pred median: {np.median(all_pred):.2f} m³/s")
        log.info(f"    Overall obs median:  {np.median(all_obs):.2f} m³/s")
        log.info(f"    Pred/Obs ratio:      {np.median(all_pred)/np.median(all_obs):.3f}")
    log.info(f"{'='*60}\n")
    return nse_list


def run_experiment(config, data_loader, label):
    """Train and evaluate, returning (pred_dict, obs_dict)."""
    from dmg.models.uh_model_handler import UhModelHandler
    from dmg.trainers.uh_trainer import UhTrainer

    log.info(f"\n>>> {label}")
    log.info(f"    Train: {config['train']['start_time']} to {config['train']['end_time']}")
    log.info(f"    Test:  {config['test']['start_time']} to {config['test']['end_time']}")

    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    # Fresh model
    model = UhModelHandler(config, verbose=False)
    model.to(config['device'])

    trainer = UhTrainer(
        config=config,
        model=model,
        data_loader=data_loader,
        verbose=False,
    )

    # Train
    t0 = time.perf_counter()
    trainer.train()
    log.info(f"    Training: {(time.perf_counter()-t0)/60:.1f} min")

    # Evaluate — collect predictions manually
    model.eval()
    pred_dict = {}
    obs_dict = {}
    warm_up = config['model']['warm_up']

    data_loader.load_dataset(mode='test')
    sampler = trainer.sampler

    for chunk_data in data_loader.get_dataset():
        n_gauges = len(chunk_data.gauge)
        eval_bs = config.get('test', {}).get('batch_size', 4)

        with torch.no_grad():
            for g_start in range(0, n_gauges, eval_bs):
                g_end = min(g_start + eval_bs, n_gauges)
                batch = sampler.get_validation_sample(chunk_data, g_start, g_end)
                batch_gpu = {
                    k: v.to(config['device']) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model(batch_gpu, eval=True)

                q_gauge = model.q_gauge.cpu()
                target = batch['target']
                gauge_indices = batch['batch_sample']

                for i, gidx in enumerate(gauge_indices):
                    gidx = int(gidx)
                    pred_dict[gidx] = q_gauge[warm_up:, i].numpy()
                    obs_dict[gidx] = target[i, warm_up:].numpy()

    # Clear test dataset cache for next experiment
    data_loader.test_dataset = None

    return pred_dict, obs_dict


def main():
    parser = argparse.ArgumentParser(description='Diagnose dHBV2.0UH pipeline')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--device', default=None, type=str)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    base_config = load_config(args.config)
    if args.device:
        base_config['device'] = args.device

    # ================================================================
    # Experiment 1: Overfit — train and test on SAME period
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT 1: Train and test on SAME period (overfit test)")
    log.info("Expected: high NSE if pipeline is correct")
    log.info("=" * 60)

    config1 = setup_config(
        {**base_config},
        output_dir='output/diagnose_exp1',
        train_start='1985/10/01',
        train_end='1987/09/30',
        test_start='1985/10/01',
        test_end='1987/09/30',
    )

    from dmg.core.data.loaders.uh_hydro_loader import UhHydroLoader
    data_loader = UhHydroLoader(config1)

    pred1, obs1 = run_experiment(config1, data_loader, "EXP 1: Overfit")
    nse1 = diagnose_predictions(pred1, obs1, "EXP 1: Train=Test period")

    # ================================================================
    # Experiment 2: Generalization — train and test on DIFFERENT periods
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT 2: Train and test on DIFFERENT periods")
    log.info("Expected: reasonable NSE if model generalizes")
    log.info("=" * 60)

    config2 = setup_config(
        {**base_config},
        output_dir='output/diagnose_exp2',
        train_start='1985/10/01',
        train_end='1987/09/30',
        test_start='1987/10/01',
        test_end='1989/09/30',
    )

    data_loader2 = UhHydroLoader(config2)
    pred2, obs2 = run_experiment(config2, data_loader2, "EXP 2: Generalization")
    nse2 = diagnose_predictions(pred2, obs2, "EXP 2: Train≠Test period")

    # ================================================================
    # Summary
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("DIAGNOSIS SUMMARY")
    log.info("=" * 60)
    if nse1:
        log.info(f"  Exp 1 (overfit):        median NSE = {np.median(nse1):.3f}")
    if nse2:
        log.info(f"  Exp 2 (generalization): median NSE = {np.median(nse2):.3f}")
    log.info("")

    if nse1 and np.median(nse1) < 0:
        log.error(
            "  ❌ Exp 1 FAILED: negative NSE on training data.\n"
            "     Pipeline bug: likely unit mismatch, wrong accumulation,\n"
            "     or target/prediction time misalignment."
        )
    elif nse1 and np.median(nse1) > 0.5:
        log.info("  ✓ Exp 1 PASSED: model can overfit training data.")
    else:
        log.warning("  ⚠ Exp 1 marginal: NSE > 0 but < 0.5. May need more epochs.")

    if nse2 and np.median(nse2) > 0:
        log.info("  ✓ Exp 2 PASSED: model generalizes to unseen period.")
    elif nse2:
        log.warning("  ⚠ Exp 2: negative NSE on test period. Normal with 10 gauges + 30 epochs.")

    log.info("=" * 60)


if __name__ == '__main__':
    main()
