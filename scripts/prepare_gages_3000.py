"""Convert gages_3000 zarr data to dMG pickle format.

Replicates NeuralHydrology's MeritHydro hourly-to-daily aggregation exactly
(timezone-aware, local standard time day boundaries), then computes Hargreaves
PET and packages everything into the pickle format that HydroLoader expects.

Usage:
    python scripts/prepare_gages_3000.py \
        --data-dir /mnt/ssd1/data/merit_hydro \
        --basin-list ~/projects/neuralhydrology/examples/merit_hydro/dhbv2_2717_basin_list.txt \
        --output-dir ./data/gages_3000
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UTC offset (matches neuralhydrology/datasetzoo/merithydro.py exactly)
# ---------------------------------------------------------------------------
def _utc_offset_hours(longitude: float) -> int:
    """Approximate UTC offset from longitude for local standard time."""
    return int(round(longitude / 15))


# ---------------------------------------------------------------------------
# Hargreaves PET (matches dhbv2/src/dhbv2/pet.py)
# ---------------------------------------------------------------------------
def hargreaves_pet(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tmean: np.ndarray,
    lat_rad: np.ndarray,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """Hargreaves PET in mm/day.

    Parameters
    ----------
    tmin, tmax, tmean : (n_days,) or (n_days, n_basins)
        Daily temperatures in degrees Celsius.
    lat_rad : (n_basins,)
        Latitude in radians.
    day_of_year : (n_days,)
        Day of year (1-366).
    """
    SOLAR_CONSTANT = 0.0820
    trange = np.maximum(tmax - tmin, 0)

    sol_dec = 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)
    sha = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(sol_dec), -1, 1))
    ird = 1 + 0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year)

    et_rad = (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT
        * ird
        * (
            sha * np.sin(lat_rad) * np.sin(sol_dec)
            + np.cos(lat_rad) * np.cos(sol_dec) * np.sin(sha)
        )
    )

    pet = 0.0023 * (tmean + 17.8) * np.sqrt(trange) * 0.408 * et_rad
    return np.maximum(pet, 0)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def main(data_dir: Path, basin_list_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load basin list ---------------------------------------------------
    basins = [
        line.strip().zfill(8)
        for line in basin_list_path.read_text().splitlines()
        if line.strip()
    ]
    n_basins = len(basins)
    log.info(f"Loaded {n_basins} basins from {basin_list_path}")

    # --- Load gage info (longitude, latitude, drainage area) ---------------
    gage_csv = pd.read_csv(data_dir / 'gage_info.csv', dtype={'STAID': str})
    gage_csv['STAID'] = gage_csv['STAID'].str.zfill(8)
    gage_csv = gage_csv.set_index('STAID')

    missing = [b for b in basins if b not in gage_csv.index]
    if missing:
        raise ValueError(f"{len(missing)} basins missing from gage_info.csv: {missing[:5]}")

    gage_info = gage_csv.loc[basins]

    # --- Load hourly AORC forcings -----------------------------------------
    log.info("Loading AORC hourly forcings from zarr...")
    aorc_ds = xarray.open_zarr(data_dir / 'aorc.zarr', consolidated=False)

    available = set(aorc_ds.gauge_id.values)
    missing_aorc = [b for b in basins if b not in available]
    if missing_aorc:
        raise ValueError(f"{len(missing_aorc)} basins missing from AORC zarr: {missing_aorc[:5]}")

    aorc_subset = aorc_ds.sel(gauge_id=basins).load()
    gauge_ids = aorc_subset.gauge_id.values
    dates_hourly = aorc_subset.date.values
    temp_hourly = aorc_subset.temperature.values      # (n_basins, n_hours) in K
    precip_hourly = aorc_subset.total_precipitation.values  # mm/hr
    n_hours = temp_hourly.shape[1]
    log.info(f"Loaded {n_basins} x {n_hours} hourly timesteps")

    # --- Aggregate hourly -> daily (NeuralHydrology algorithm) -------------
    log.info("Aggregating to daily using local standard time boundaries...")
    utc_offsets = np.array([
        _utc_offset_hours(gage_info.loc[b, 'LNG_GAGE']) for b in gauge_ids
    ])
    max_abs_offset = int(max(abs(o) for o in utc_offsets))
    n_days = (n_hours - max_abs_offset) // 24

    temp_mean_daily = np.empty((n_basins, n_days), dtype=np.float32)
    temp_min_daily = np.empty((n_basins, n_days), dtype=np.float32)
    temp_max_daily = np.empty((n_basins, n_days), dtype=np.float32)
    precip_daily = np.empty((n_basins, n_days), dtype=np.float32)

    for offset in np.unique(utc_offsets):
        mask = utc_offsets == offset
        shift = abs(offset)
        end = shift + n_days * 24

        t = temp_hourly[mask, shift:end]
        p = precip_hourly[mask, shift:end]

        n_group = int(mask.sum())
        t_reshaped = t.reshape(n_group, n_days, 24)
        p_reshaped = p.reshape(n_group, n_days, 24)

        temp_mean_daily[mask] = np.nanmean(t_reshaped, axis=2)
        temp_min_daily[mask] = np.nanmin(t_reshaped, axis=2)
        temp_max_daily[mask] = np.nanmax(t_reshaped, axis=2)
        precip_daily[mask] = np.nansum(p_reshaped, axis=2)

    dates_daily = pd.DatetimeIndex(
        dates_hourly[max_abs_offset::24][:n_days]
    ).normalize()

    del aorc_subset, temp_hourly, precip_hourly

    # --- Fill NaN forcings with day-of-year climatology -----------------------
    # Matches NeuralHydrology MeritHydro._load_basin_data(): per-basin, group by
    # dayofyear, compute mean across all years, fillna with that climatology.
    doy = dates_daily.dayofyear.values  # (n_days,)
    unique_doys = np.unique(doy)

    nan_before = sum(
        np.isnan(a).sum()
        for a in [temp_mean_daily, temp_min_daily, temp_max_daily, precip_daily]
    )

    for arr in [temp_mean_daily, temp_min_daily, temp_max_daily, precip_daily]:
        for d in unique_doys:
            day_mask = doy == d
            subset = arr[:, day_mask]                    # (n_basins, n_years_for_doy)
            clim = np.nanmean(subset, axis=1, keepdims=True)  # (n_basins, 1)
            nans = np.isnan(subset)
            if nans.any():
                subset[nans] = np.broadcast_to(clim, subset.shape)[nans]
                arr[:, day_mask] = subset

    nan_after = sum(
        np.isnan(a).sum()
        for a in [temp_mean_daily, temp_min_daily, temp_max_daily, precip_daily]
    )
    log.info(f"Filled NaN forcings with DOY climatology: {nan_before} -> {nan_after} remaining")

    # Convert temperature K -> C
    temp_mean_c = temp_mean_daily - 273.15
    temp_min_c = temp_min_daily - 273.15
    temp_max_c = temp_max_daily - 273.15

    log.info(f"Daily aggregation: {n_days} days, {dates_daily[0]} to {dates_daily[-1]}")

    # --- Compute Hargreaves PET --------------------------------------------
    log.info("Computing Hargreaves PET...")
    lat_deg = gage_info['LAT_GAGE'].values.astype(np.float32)
    lat_rad = np.deg2rad(lat_deg)  # (n_basins,)

    day_of_year = dates_daily.dayofyear.values  # (n_days,)

    # Broadcast: lat_rad (n_basins,) -> (n_basins, 1), doy (n_days,) -> (1, n_days)
    lat_bc = lat_rad[:, np.newaxis]      # (n_basins, 1)
    doy_bc = day_of_year[np.newaxis, :]  # (1, n_days)

    pet_daily = hargreaves_pet(
        tmin=temp_min_c,
        tmax=temp_max_c,
        tmean=temp_mean_c,
        lat_rad=lat_bc,
        day_of_year=doy_bc,
    ).astype(np.float32)

    # --- Load static attributes from zarr ----------------------------------
    log.info("Loading static attributes from zarr...")
    attrs_ds = xarray.open_zarr(data_dir / 'gage_merit_attrs.zarr', consolidated=False)
    attrs_df = attrs_ds.to_dataframe()
    attrs_df.index = attrs_df.index.astype(str).str.zfill(8)
    attrs_df.index.name = 'STAID'

    # The 14 attributes matching NeuralHydrology LSTM (order must match
    # all_attributes in observation YAML).
    attr_names = [
        'DRAIN_SQKM', 'meanP', 'ETPOT_Hargr', 'aridity', 'seasonality_P',
        'snow_fraction', 'meanelevation', 'meanslope', 'NDVI', 'Porosity',
        'HWSD_sand', 'HWSD_silt', 'HWSD_clay', 'permeability',
    ]

    # DRAIN_SQKM comes from gage_info.csv, rest from zarr.
    attr_arrays = []
    for name in attr_names:
        if name == 'DRAIN_SQKM':
            attr_arrays.append(gage_info['DRAIN_SQKM'].values.astype(np.float32))
        elif name in attrs_df.columns:
            attr_arrays.append(
                attrs_df.loc[basins, name].values.astype(np.float32)
            )
        else:
            raise ValueError(f"Attribute '{name}' not found in zarr or gage_info")

    attributes = np.column_stack(attr_arrays)  # (n_basins, 14)

    # --- Load USGS daily streamflow ----------------------------------------
    log.info("Loading USGS streamflow from icechunk store...")
    try:
        import icechunk
    except ImportError:
        raise ImportError("icechunk package required: pip install icechunk")

    sf_store = data_dir / 'usgs_streamflow'
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(sf_store))
    )
    session = repo.readonly_session('main')
    sf_ds = xarray.open_zarr(session.store, consolidated=False)

    sf_subset = sf_ds.sel(gage_id=basins).load()
    sf_time = pd.DatetimeIndex(sf_subset.time.values).normalize()

    # streamflow array: dims are (gage_id, time) -> (n_basins, n_sf_days)
    sf_data = sf_subset['streamflow'].values  # (n_basins, n_sf_days)
    log.info(f"Streamflow shape: {sf_data.shape}, time range: {sf_time[0]} to {sf_time[-1]}")

    # Find common dates between forcings and streamflow
    common_dates = dates_daily.intersection(sf_time)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between forcings and streamflow")

    # Indices into each array for the common dates
    forc_idx = np.array([dates_daily.get_loc(d) for d in common_dates])
    sf_idx = np.array([sf_time.get_loc(d) for d in common_dates])

    dates_final = common_dates
    n_days_final = len(dates_final)

    # Subset all arrays to common date range
    temp_mean_c = temp_mean_c[:, forc_idx]
    precip_daily = precip_daily[:, forc_idx]
    pet_daily = pet_daily[:, forc_idx]
    target_vals = sf_data[:, sf_idx].astype(np.float32)  # (n_basins, n_days)

    log.info(
        f"Final dataset: {n_basins} basins x {n_days_final} days "
        f"({dates_final[0]} to {dates_final[-1]})"
    )

    # --- Build pickle arrays -----------------------------------------------
    # HydroLoader expects: (forcings, target, attributes)
    # forcings: (n_basins, n_days, n_forcing_vars)
    # target:   (n_basins, n_days, 1)
    # attributes: (n_basins, n_attributes)
    forcings = np.stack([precip_daily, temp_mean_c, pet_daily], axis=2)
    target = target_vals[:, :, np.newaxis]

    log.info(
        f"Pickle shapes: forcings={forcings.shape}, "
        f"target={target.shape}, attributes={attributes.shape}"
    )

    # --- Save outputs ------------------------------------------------------
    pkl_path = output_dir / 'gages_3000_daily.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump((forcings, target, attributes), f, protocol=4)
    log.info(f"Saved pickle: {pkl_path}")

    gage_id_path = output_dir / 'gages_3000_gage_id.npy'
    np.save(gage_id_path, np.array(basins))
    log.info(f"Saved gage IDs: {gage_id_path}")

    # Save date index for reference
    dates_path = output_dir / 'gages_3000_dates.npy'
    np.save(dates_path, dates_final.values)
    log.info(f"Saved dates: {dates_path}")

    log.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='Convert gages_3000 to dMG pickle')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/mnt/ssd1/data/merit_hydro'),
        help='Root data directory with aorc.zarr, gage_info.csv, etc.',
    )
    parser.add_argument(
        '--basin-list',
        type=Path,
        default=Path.home() / 'projects/neuralhydrology/examples/merit_hydro/dhbv2_2717_basin_list.txt',
        help='Path to basin list file (one STAID per line).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data/gages_3000'),
        help='Directory for output pickle and metadata files.',
    )
    args = parser.parse_args()

    main(args.data_dir, args.basin_list, args.output_dir)
