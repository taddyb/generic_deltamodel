"""Forward trained dHBV2 lumped on MERIT unit catchments to produce DDR lateral inflows.

Loads a trained dHBV2 (LstmMlpModel + Hbv_2), runs inference on ~288K MERIT unit
catchments using AORC hourly forcings, and writes Qr (m³/s) to an icechunk store
in DDR-compatible format.

Usage:
    python scripts/forward_merit_dhbv2.py \
        --model-dir <path_to_model_checkpoints> \
        --epoch 100 \
        --output /mnt/ssd1/data/icechunk/daily_dhbv2_merit_unit_catchments.ic

Requirements:
    - AORC forcings: /mnt/ssd1/data/aorc/merit_unit_catchments.zarr
    - MERIT attrs: ~/projects/ddr/data/merit_global_attributes_v2.nc
    - icechunk package
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger(__name__)

MERIT_ZONES = ['71', '72', '73', '74', '75', '77', '78']
ZONE_UTC_OFFSETS = {
    '71': -5, '72': -5, '73': -5, '74': -6,
    '75': -6, '77': -7, '78': -8,
}

# Static attribute mapping: training name → MERIT global attr name
ATTR_MAP = {
    'DRAIN_SQKM': 'log10_uparea',  # special: 10^log10_uparea
    'meanP': 'meanP',
    'ETPOT_Hargr': 'ETPOT_Hargr',
    'aridity': 'aridity',
    'seasonality_P': 'seasonality_P',
    'snow_fraction': 'snow_fraction',
    'meanelevation': 'meanelevation',
    'meanslope': 'meanslope',
    'NDVI': 'NDVI',
    'Porosity': 'Porosity',
    'HWSD_sand': 'HWSD_sand',
    'HWSD_silt': 'HWSD_silt',
    'HWSD_clay': 'HWSD_clay',
    'permeability': 'permeability',
}

# Forcing and attribute names (must match training config order)
NN_FORCINGS = ['prcp', 'tmean']
PHY_FORCINGS = ['prcp', 'tmean', 'pet']
NN_ATTRIBUTES = [
    'DRAIN_SQKM', 'meanP', 'ETPOT_Hargr', 'aridity', 'seasonality_P',
    'snow_fraction', 'meanelevation', 'meanslope', 'NDVI', 'Porosity',
    'HWSD_sand', 'HWSD_silt', 'HWSD_clay', 'permeability',
]


# ---------------------------------------------------------------------------
# Hargreaves PET (matches dhbv2/src/dhbv2/pet.py)
# ---------------------------------------------------------------------------
def hargreaves_pet(tmin, tmax, tmean, lat_rad, day_of_year):
    """Hargreaves PET in mm/day."""
    SOLAR_CONSTANT = 0.0820
    trange = np.maximum(tmax - tmin, 0)
    sol_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    sha = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(sol_dec), -1, 1))
    ird = 1 + 0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year)
    et_rad = (
        (24.0 * 60.0 / np.pi) * SOLAR_CONSTANT * ird
        * (sha * np.sin(lat_rad) * np.sin(sol_dec)
           + np.cos(lat_rad) * np.cos(sol_dec) * np.sin(sha))
    )
    pet = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * et_rad
    return np.maximum(pet, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_basin_list(aorc_zarr_path: str) -> np.ndarray:
    """Load basin COMIDs from AORC zarr, filtered to MERIT zones."""
    ds = xr.open_zarr(aorc_zarr_path, consolidated=False)
    all_ids = ds.gauge_id.values
    mask = np.array([str(gid)[:2] in MERIT_ZONES for gid in all_ids])
    filtered = all_ids[mask]
    LOG.info(f"Filtered {len(all_ids)} -> {len(filtered)} basins (zones {', '.join(MERIT_ZONES)})")
    return filtered


def aggregate_hourly_to_daily(temp_h, precip_h, comids, dates_h):
    """Aggregate hourly forcings to daily using zone-based timezone offsets.

    Also returns Tmin and Tmax for Hargreaves PET computation.
    """
    n_basins, n_hours = temp_h.shape
    zones = np.array([str(c)[:2] for c in comids])

    max_abs_offset = max(abs(ZONE_UTC_OFFSETS[z]) for z in MERIT_ZONES)
    n_days = (n_hours - max_abs_offset) // 24

    temp_mean = np.empty((n_basins, n_days), dtype=np.float32)
    temp_min = np.empty((n_basins, n_days), dtype=np.float32)
    temp_max = np.empty((n_basins, n_days), dtype=np.float32)
    precip_daily = np.empty((n_basins, n_days), dtype=np.float32)

    for zone, offset in ZONE_UTC_OFFSETS.items():
        mask = zones == zone
        if not mask.any():
            continue
        shift = abs(offset)
        end = shift + n_days * 24
        n_group = int(mask.sum())

        t_reshaped = temp_h[mask, shift:end].reshape(n_group, n_days, 24)
        p_reshaped = precip_h[mask, shift:end].reshape(n_group, n_days, 24)

        temp_mean[mask] = np.nanmean(t_reshaped, axis=2)
        temp_min[mask] = np.nanmin(t_reshaped, axis=2)
        temp_max[mask] = np.nanmax(t_reshaped, axis=2)
        precip_daily[mask] = np.nansum(p_reshaped, axis=2)

    dates_daily = pd.DatetimeIndex(dates_h[max_abs_offset::24][:n_days]).normalize().values
    return temp_mean, temp_min, temp_max, precip_daily, dates_daily


def fill_nan_climatology(data, dates):
    """Fill NaN with day-of-year climatology per basin."""
    doy = pd.DatetimeIndex(dates).dayofyear.values
    unique_doys = np.unique(doy)
    for d in unique_doys:
        day_mask = doy == d
        subset = data[:, day_mask]
        clim = np.nanmean(subset, axis=1, keepdims=True)
        nans = np.isnan(subset)
        if nans.any():
            subset[nans] = np.broadcast_to(clim, subset.shape)[nans]
            data[:, day_mask] = subset
    return data


def load_merit_attributes(attrs_path, comids):
    """Load raw (unnormalized) static attributes for given COMIDs."""
    ds = xr.open_dataset(attrs_path)
    int_comids = np.array([int(c) for c in comids])

    n_basins = len(comids)
    n_attrs = len(NN_ATTRIBUTES)
    attrs = np.empty((n_basins, n_attrs), dtype=np.float32)

    for i, attr_name in enumerate(NN_ATTRIBUTES):
        merit_name = ATTR_MAP[attr_name]
        raw = ds[merit_name].sel(COMID=int_comids).values.astype(np.float32)
        if attr_name == 'DRAIN_SQKM':
            raw = np.power(10, raw)  # log10_uparea -> km²
        attrs[:, i] = raw

    return attrs


def normalize_array(data, stats, var_names):
    """Normalize data using dmg norm stats [p10, p90, mean, std]."""
    out = np.empty_like(data, dtype=np.float32)
    for i, name in enumerate(var_names):
        mean, std = stats[name][2], stats[name][3]
        out[..., i] = (data[..., i] - mean) / std
    return out


def mm_day_to_m3s(q_mm_day, catchsize_km2):
    """Convert mm/day to m³/s using local catchment area."""
    conversion = catchsize_km2[:, np.newaxis] * 1000 / 86400
    qr = q_mm_day * conversion
    qr = np.nan_to_num(qr, nan=1e-6, posinf=1e-6, neginf=1e-6)
    qr[qr <= 0] = 1e-6
    return qr.astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_dhbv2_model(model_dir: Path, config_path: Path, epoch: int,
                     device: torch.device):
    """Load trained dHBV2 model from checkpoint."""
    from omegaconf import OmegaConf

    from dmg.core.utils.utils import initialize_config
    from dmg.models.model_handler import ModelHandler

    # Load config
    raw_config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw_config, resolve=True)

    # Override model_dir and test_epoch
    config['model_dir'] = str(model_dir) + '/'
    config['test'] = config.get('test', {})
    config['test']['test_epoch'] = epoch
    config['device'] = str(device)
    config['mode'] = 'test'

    config = initialize_config(config, make_dirs=False, write_out=False)

    model = ModelHandler(config, verbose=True)
    model.eval()
    LOG.info(f"Loaded dHBV2 from {model_dir}, epoch {epoch}")
    return model, config


# ---------------------------------------------------------------------------
# Main forward pass
# ---------------------------------------------------------------------------
def forward_daily(model_dir: Path, config_path: Path, epoch: int,
                  aorc_path: str, attrs_path: str, output_path: str,
                  device: torch.device, cpu_batch_size: int, gpu_batch_size: int):
    """Forward dHBV2 lumped on all MERIT basins, daily mode."""
    import icechunk
    import zarr

    # Load model and norm stats
    model, config = load_dhbv2_model(model_dir, config_path, epoch, device)

    norm_stats_path = model_dir / 'normalization_statistics.json'
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    comids = get_basin_list(aorc_path)
    n_basins = len(comids)
    int_comids = np.array([int(c) for c in comids])

    # Open AORC lazily
    aorc_ds = xr.open_zarr(aorc_path, consolidated=False)
    dates_hourly = aorc_ds.date.values

    # Load catchsize for mm/day -> m³/s and for ac_all
    attrs_ds = xr.open_dataset(attrs_path)
    all_catchsize = attrs_ds['catchsize'].sel(COMID=int_comids).values.astype(np.float32)

    # Determine time dimension
    max_abs_offset = max(abs(ZONE_UTC_OFFSETS[z]) for z in MERIT_ZONES)
    n_hours = len(dates_hourly)
    n_days_full = (n_hours - max_abs_offset) // 24
    dates_daily_full = pd.DatetimeIndex(
        dates_hourly[max_abs_offset::24][:n_days_full]
    ).normalize()

    # Warmup: discard 1980 — output starts 1981-01-01
    warmup_cutoff = pd.Timestamp('1981-01-01')
    warmup_days = int((dates_daily_full < warmup_cutoff).sum())
    dates_output = dates_daily_full[warmup_days:].values
    n_days_output = len(dates_output)

    LOG.info(f"Full series: {n_days_full} days, warmup: {warmup_days} days (1980)")
    LOG.info(f"Output: {n_basins} basins x {n_days_output} days "
             f"({dates_output[0]} to {dates_output[-1]})")

    # Create output icechunk store
    storage = icechunk.local_filesystem_storage(output_path)
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session('main')

    chunk_basins = min(3080, n_basins)
    chunk_time = min(468, n_days_output)

    out_ds = xr.Dataset(
        data_vars={
            'Qr': (['divide_id', 'time'],
                    np.full((n_basins, n_days_output), 1e-6, dtype=np.float32),
                    {'units': 'm^3/s'}),
        },
        coords={
            'divide_id': ('divide_id', int_comids),
            'time': ('time', dates_output),
        },
        attrs={'units': 'm^3/s', 'source': 'dHBV2 lumped runoff'},
    )
    out_ds.to_zarr(session.store, mode='w', encoding={
        'Qr': {'chunks': (chunk_basins, chunk_time)},
    })
    session.commit('Initialize store')

    # Warm up period for model (rho)
    rho = config['model']['rho']  # 365

    t0 = time.time()
    for batch_start in range(0, n_basins, cpu_batch_size):
        batch_end = min(batch_start + cpu_batch_size, n_basins)
        batch_comids = comids[batch_start:batch_end]
        batch_size = batch_end - batch_start
        batch_catchsize = all_catchsize[batch_start:batch_end]

        LOG.info(f"Batch {batch_start // cpu_batch_size + 1}/"
                 f"{(n_basins + cpu_batch_size - 1) // cpu_batch_size}: "
                 f"basins {batch_start}-{batch_end}")

        # 1. Load hourly AORC for batch
        batch_aorc = aorc_ds.sel(gauge_id=batch_comids).load()
        temp_h = batch_aorc.temperature.values
        precip_h = batch_aorc.total_precipitation.values
        del batch_aorc

        # 2. Aggregate to daily (with Tmin/Tmax for PET)
        temp_mean, temp_min, temp_max, precip_d, _ = aggregate_hourly_to_daily(
            temp_h, precip_h, batch_comids, dates_hourly)
        del temp_h, precip_h

        # 3. Fill NaN with DOY climatology
        temp_mean = fill_nan_climatology(temp_mean, dates_daily_full.values)
        temp_min = fill_nan_climatology(temp_min, dates_daily_full.values)
        temp_max = fill_nan_climatology(temp_max, dates_daily_full.values)
        precip_d = fill_nan_climatology(precip_d, dates_daily_full.values)

        # 4. Convert K -> C
        temp_mean_c = temp_mean - 273.15
        temp_min_c = temp_min - 273.15
        temp_max_c = temp_max - 273.15

        # 5. Compute Hargreaves PET
        # Get latitude from MERIT attributes
        batch_int_comids = np.array([int(c) for c in batch_comids])
        lat_deg = attrs_ds['meanelevation'].sel(COMID=batch_int_comids).values  # placeholder
        # Actually need latitude — use a simple approximation from meanelevation
        # or load from a latitude variable if available
        # For MERIT, latitude is typically stored as 'lat' or computed from COMID
        # Try to get latitude from the attributes dataset
        if 'lat' in attrs_ds:
            lat_deg = attrs_ds['lat'].sel(COMID=batch_int_comids).values.astype(np.float32)
        elif 'latitude' in attrs_ds:
            lat_deg = attrs_ds['latitude'].sel(COMID=batch_int_comids).values.astype(np.float32)
        else:
            # Approximate from ETPOT_Hargr — back-compute rough latitude
            # Or use a fixed mid-CONUS latitude as fallback
            LOG.warning("No latitude variable in attrs — using CONUS mean (39°N)")
            lat_deg = np.full(batch_size, 39.0, dtype=np.float32)

        lat_rad = np.deg2rad(lat_deg)[:, np.newaxis]
        doy = pd.DatetimeIndex(dates_daily_full).dayofyear.values[np.newaxis, :]
        pet_d = hargreaves_pet(temp_min_c, temp_max_c, temp_mean_c, lat_rad, doy)

        # 6. Load and normalize static attributes
        raw_attrs = load_merit_attributes(attrs_path, batch_comids)

        # ac_all = DRAIN_SQKM (first attribute), elev_all = meanelevation (7th)
        ac_all = raw_attrs[:, NN_ATTRIBUTES.index('DRAIN_SQKM')]
        elev_all = raw_attrs[:, NN_ATTRIBUTES.index('meanelevation')]

        # Normalize attributes
        attrs_norm = np.empty_like(raw_attrs)
        for i, name in enumerate(NN_ATTRIBUTES):
            mean, std = norm_stats[name][2], norm_stats[name][3]
            attrs_norm[:, i] = (raw_attrs[:, i] - mean) / std

        # 7. Build normalized NN forcings: [time, basins, 2] (prcp, tmean only)
        prcp_mean, prcp_std = norm_stats['prcp'][2], norm_stats['prcp'][3]
        tmean_mean, tmean_std = norm_stats['tmean'][2], norm_stats['tmean'][3]

        prcp_norm = (precip_d - prcp_mean) / prcp_std       # (basins, days)
        tmean_norm = (temp_mean_c - tmean_mean) / tmean_std  # (basins, days)

        # 8. GPU sub-batching
        all_preds = np.empty((batch_size, n_days_output), dtype=np.float32)

        for sub_start in range(0, batch_size, gpu_batch_size):
            sub_end = min(sub_start + gpu_batch_size, batch_size)
            sub_size = sub_end - sub_start

            with torch.no_grad():
                # x_phy: [time, sub, 3] (prcp, tmean_C, pet) — unnormalized for physics
                x_phy = torch.from_numpy(np.stack([
                    precip_d[sub_start:sub_end].T,      # (days, sub)
                    temp_mean_c[sub_start:sub_end].T,
                    pet_d[sub_start:sub_end].T,
                ], axis=2).astype(np.float32)).to(device)

                # xc_nn_norm: [time, sub, n_forcings + n_attrs] — normalized
                # NN forcings: prcp, tmean (2 vars)
                x_nn_t = np.stack([
                    prcp_norm[sub_start:sub_end].T,
                    tmean_norm[sub_start:sub_end].T,
                ], axis=2)  # (days, sub, 2)

                # Expand static attrs across time
                c_nn_t = np.broadcast_to(
                    attrs_norm[sub_start:sub_end][np.newaxis, :, :],
                    (n_days_full, sub_size, len(NN_ATTRIBUTES)),
                ).copy()

                xc_nn_norm = torch.from_numpy(
                    np.concatenate([x_nn_t, c_nn_t], axis=2).astype(np.float32)
                ).to(device)

                # c_nn_norm: [sub, n_attrs] — for MLP head
                c_nn_norm = torch.from_numpy(
                    attrs_norm[sub_start:sub_end].astype(np.float32)
                ).to(device)

                # ac_all, elev_all: [sub]
                ac = torch.from_numpy(
                    ac_all[sub_start:sub_end].astype(np.float32)
                ).to(device)
                elev = torch.from_numpy(
                    elev_all[sub_start:sub_end].astype(np.float32)
                ).to(device)

                dataset_sample = {
                    'x_phy': x_phy,
                    'xc_nn_norm': xc_nn_norm,
                    'c_nn_norm': c_nn_norm,
                    'ac_all': ac,
                    'elev_all': elev,
                }

                output = model(dataset_sample, eval=True)

                # Extract streamflow prediction
                model_name = list(output.keys())[0]
                pred = output[model_name]['streamflow']  # [time, sub, 1]
                pred = pred[warmup_days:, :, 0].cpu().numpy()  # [n_days_output, sub]
                all_preds[sub_start:sub_end] = pred.T  # (sub, n_days_output)

                del dataset_sample, output, pred
                torch.cuda.empty_cache()

        # 9. Convert mm/day -> m³/s using catchsize (NOT DRAIN_SQKM)
        qr = mm_day_to_m3s(all_preds, batch_catchsize)

        # 10. Write batch to icechunk store
        session = repo.writable_session('main')
        root = zarr.open_group(session.store, mode='r+')
        root['Qr'][batch_start:batch_end, :] = qr
        session.commit(f'Batch {batch_start}-{batch_end}')

        elapsed = time.time() - t0
        rate = batch_end / elapsed
        eta = (n_basins - batch_end) / rate if rate > 0 else 0
        LOG.info(f"  Written. Rate: {rate:.0f} basins/s, ETA: {eta / 60:.1f} min")

    LOG.info(f"Done. {n_basins} basins in {(time.time() - t0) / 60:.1f} min")
    LOG.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Forward dHBV2 lumped on MERIT unit catchments')
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='Path to model checkpoint directory')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent.parent / 'conf' / 'config_dhbv_2_lumped.yaml',
                        help='Path to dHBV2 Hydra config YAML')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Model epoch to load')
    parser.add_argument('--output', default=None,
                        help='Output icechunk store path')
    parser.add_argument('--aorc',
                        default='/mnt/ssd1/data/aorc/merit_unit_catchments.zarr',
                        help='AORC forcings zarr path')
    parser.add_argument('--attrs',
                        default='/home/tbindas/projects/ddr/data/merit_global_attributes_v2.nc',
                        help='MERIT global attributes path')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--cpu-batch', type=int, default=5000,
                        help='CPU batch size (basins loaded into RAM at once)')
    parser.add_argument('--gpu-batch', type=int, default=100,
                        help='GPU sub-batch size (basins forwarded at once)')
    args = parser.parse_args()

    output = args.output or '/mnt/ssd1/data/icechunk/daily_dhbv2_merit_unit_catchments.ic'
    device = torch.device(args.device)

    forward_daily(
        model_dir=args.model_dir,
        config_path=args.config,
        epoch=args.epoch,
        aorc_path=args.aorc,
        attrs_path=args.attrs,
        output_path=output,
        device=device,
        cpu_batch_size=args.cpu_batch,
        gpu_batch_size=args.gpu_batch,
    )


if __name__ == '__main__':
    main()
