"""Regression tests for pure helper functions in forward_merit_dhbv2.py.

These tests snapshot the current behavior of each function so that
performance optimizations don't silently break correctness.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import helpers from the script (scripts/ is not a package).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'forward_merit_dhbv2.py'
_spec = importlib.util.spec_from_file_location('forward_merit_dhbv2', _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

hargreaves_pet = _mod.hargreaves_pet
aggregate_hourly_to_daily = _mod.aggregate_hourly_to_daily
fill_nan_climatology = _mod.fill_nan_climatology
normalize_array = _mod.normalize_array
mm_day_to_m3s = _mod.mm_day_to_m3s


# ===================================================================
# TestHargreavesPet
# ===================================================================
class TestHargreavesPet:
    """Hargreaves PET: tmin/tmax/tmean (°C), lat_rad, day_of_year -> mm/day."""

    def test_known_value_40n_solstice(self):
        lat_rad = np.deg2rad(np.array([[40.0]]))
        pet = hargreaves_pet(
            np.array([[15.0]]),
            np.array([[30.0]]),
            np.array([[22.5]]),
            lat_rad,
            np.array([[172]]),
        )
        np.testing.assert_allclose(pet[0, 0], 6.1329455, rtol=1e-5)

    def test_shape_broadcast(self):
        np.random.seed(42)
        n_basins, n_days = 3, 5
        tmin = np.random.uniform(5, 15, (n_basins, n_days)).astype(np.float32)
        tmax = tmin + np.random.uniform(5, 15, (n_basins, n_days)).astype(np.float32)
        tmean = (tmin + tmax) / 2
        lats = np.deg2rad(np.array([35.0, 40.0, 45.0])[:, np.newaxis]).astype(np.float32)
        doys = np.array([100, 150, 200, 250, 300])[np.newaxis, :].astype(np.int32)
        pet = hargreaves_pet(tmin, tmax, tmean, lats, doys)
        assert pet.shape == (3, 5)

    def test_non_negativity(self):
        np.random.seed(42)
        tmin = np.random.uniform(-10, 20, (10, 30)).astype(np.float32)
        tmax = tmin + np.random.uniform(0, 20, (10, 30)).astype(np.float32)
        tmean = (tmin + tmax) / 2
        lats = np.deg2rad(np.random.uniform(25, 50, (10, 1))).astype(np.float32)
        doys = np.arange(1, 31)[np.newaxis, :].astype(np.int32)
        pet = hargreaves_pet(tmin, tmax, tmean, lats, doys)
        assert (pet >= 0).all()

    def test_dtype_float32(self):
        pet = hargreaves_pet(
            np.array([[10.0]]),
            np.array([[25.0]]),
            np.array([[17.5]]),
            np.array([[0.7]]),
            np.array([[180]]),
        )
        assert pet.dtype == np.float32

    def test_nan_propagation(self):
        pet = hargreaves_pet(
            np.array([[10.0, 10.0]]),
            np.array([[25.0, 25.0]]),
            np.array([[17.5, np.nan]]),
            np.array([[0.7]]),
            np.array([[180, 180]]),
        )
        assert not np.isnan(pet[0, 0])
        assert np.isnan(pet[0, 1])

    def test_tmin_gt_tmax_zero(self):
        pet = hargreaves_pet(
            np.array([[30.0]]),  # tmin > tmax
            np.array([[15.0]]),
            np.array([[22.5]]),
            np.deg2rad(np.array([[40.0]])),
            np.array([[172]]),
        )
        assert pet[0, 0] == 0.0

    def test_regression_2d(self):
        np.random.seed(42)
        n_basins, n_days = 3, 5
        tmin = np.random.uniform(5, 15, (n_basins, n_days)).astype(np.float32)
        tmax = tmin + np.random.uniform(5, 15, (n_basins, n_days)).astype(np.float32)
        tmean = (tmin + tmax) / 2
        lats = np.deg2rad(np.array([35.0, 40.0, 45.0])[:, np.newaxis]).astype(np.float32)
        doys = np.array([100, 150, 200, 250, 300])[np.newaxis, :].astype(np.int32)
        pet = hargreaves_pet(tmin, tmax, tmean, lats, doys)
        np.testing.assert_allclose(pet[0, 0], 2.5703933, rtol=1e-5)
        np.testing.assert_allclose(pet[1, 2], 3.7829087, rtol=1e-5)
        np.testing.assert_allclose(pet[2, 4], 1.0099839, rtol=1e-5)


# ===================================================================
# TestAggregateHourlyToDaily
# ===================================================================
class TestAggregateHourlyToDaily:
    """Hourly -> daily aggregation with timezone-aware shifting."""

    @pytest.fixture()
    def hourly_data(self):
        """Deterministic hourly data for 6 basins across 3 zones."""
        n_basins = 6
        comids = np.array([7100001, 7100002, 7400001, 7400002, 7800001, 7800002])
        n_hours = 80
        dates_h = pd.date_range('2020-01-01', periods=n_hours, freq='h').values

        # temp_h[b,h] = 280 + b*2 + (h%24)*0.5
        temp_h = np.array(
            [[280.0 + b * 2 + (h % 24) * 0.5 for h in range(n_hours)]
             for b in range(n_basins)],
            dtype=np.float32,
        )
        # precip: 0.1*(b+1) every 6th hour, else 0
        precip_h = np.zeros((n_basins, n_hours), dtype=np.float32)
        for b in range(n_basins):
            precip_h[b, ::6] = 0.1 * (b + 1)

        return temp_h, precip_h, comids, dates_h

    def test_output_shapes(self, hourly_data):
        tmean, tmin, tmax, precip_d, dates_d = aggregate_hourly_to_daily(*hourly_data)
        assert tmean.shape == (6, 3)
        assert tmin.shape == (6, 3)
        assert tmax.shape == (6, 3)
        assert precip_d.shape == (6, 3)
        assert len(dates_d) == 3

    def test_n_days_formula(self, hourly_data):
        tmean, _, _, _, _ = aggregate_hourly_to_daily(*hourly_data)
        # n_days = (80 - 8) // 24 = 3  (max_abs_offset = 8 for zone 78)
        assert tmean.shape[1] == 3

    def test_zone71_shift(self, hourly_data):
        tmean, _, _, _, _ = aggregate_hourly_to_daily(*hourly_data)
        # Zone 71, offset=-5, shift=5: basin 0 day 0 mean
        np.testing.assert_allclose(tmean[0, 0], 285.75, rtol=1e-5)

    def test_zone74_shift(self, hourly_data):
        tmean, _, _, _, _ = aggregate_hourly_to_daily(*hourly_data)
        # Zone 74, offset=-6, shift=6: basin 2 (b=2 -> +4K base)
        np.testing.assert_allclose(tmean[2, 0], 289.75, rtol=1e-5)

    def test_zone78_shift(self, hourly_data):
        tmean, _, _, _, _ = aggregate_hourly_to_daily(*hourly_data)
        # Zone 78, offset=-8, shift=8: basin 4 (b=4 -> +8K base)
        np.testing.assert_allclose(tmean[4, 0], 293.75, rtol=1e-5)

    def test_temp_min_max(self, hourly_data):
        _, tmin, tmax, _, _ = aggregate_hourly_to_daily(*hourly_data)
        np.testing.assert_allclose(tmin[0, 0], 280.0, rtol=1e-5)
        np.testing.assert_allclose(tmax[0, 0], 291.5, rtol=1e-5)

    def test_precip_is_nansum(self, hourly_data):
        _, _, _, precip_d, _ = aggregate_hourly_to_daily(*hourly_data)
        np.testing.assert_allclose(precip_d[0, 0], 0.4, rtol=1e-5)
        np.testing.assert_allclose(precip_d[4, 0], 2.0, rtol=1e-5)

    def test_dtype_float32(self, hourly_data):
        tmean, tmin, tmax, precip_d, _ = aggregate_hourly_to_daily(*hourly_data)
        assert tmean.dtype == np.float32
        assert tmin.dtype == np.float32
        assert tmax.dtype == np.float32
        assert precip_d.dtype == np.float32


# ===================================================================
# TestFillNanClimatology
# ===================================================================
class TestFillNanClimatology:
    """NaN fill with per-basin day-of-year climatology."""

    @pytest.fixture()
    def clim_data(self):
        """3 basins, 731 days (2020-01-01 to 2021-12-31)."""
        np.random.seed(42)
        n_basins, n_days = 3, 731
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D').values
        data = np.random.rand(n_basins, n_days).astype(np.float32)
        return data, dates

    def test_nan_filled_with_climatology(self, clim_data):
        data, dates = clim_data
        original_jan1_val = data[0, 0].copy()  # Only non-NaN Jan 1 value
        data[0, 366] = np.nan  # Jan 1 of 2021
        result = fill_nan_climatology(data, dates)
        # Climatology = mean of non-NaN Jan 1 values = just data[0, 0]
        np.testing.assert_allclose(result[0, 366], original_jan1_val, rtol=1e-5)

    def test_modifies_in_place(self, clim_data):
        data, dates = clim_data
        data[0, 0] = np.nan
        result = fill_nan_climatology(data, dates)
        assert result is data

    def test_no_nan_passthrough(self, clim_data):
        data, dates = clim_data
        original = data.copy()
        result = fill_nan_climatology(data, dates)
        np.testing.assert_array_equal(result, original)

    def test_all_nan_doy_stays_nan(self, clim_data):
        data, dates = clim_data
        doy = pd.DatetimeIndex(dates).dayofyear.values
        # Set ALL Jan 1 values to NaN (indices where DOY == 1)
        jan1_mask = doy == 1
        data[:, jan1_mask] = np.nan
        result = fill_nan_climatology(data, dates)
        assert np.isnan(result[:, jan1_mask]).all()

    def test_shape_preserved(self, clim_data):
        data, dates = clim_data
        original_shape = data.shape
        data[1, 100] = np.nan
        result = fill_nan_climatology(data, dates)
        assert result.shape == original_shape


# ===================================================================
# TestNormalizeArray
# ===================================================================
class TestNormalizeArray:
    """Normalize with (data - mean) / std using [p10, p90, mean, std] stats."""

    def test_known_values(self):
        data = np.array([[10.0, 20.0], [5.0, 30.0]])
        stats = {'a': [0, 0, 10.0, 5.0], 'b': [0, 0, 20.0, 10.0]}
        out = normalize_array(data, stats, ['a', 'b'])
        np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-7)   # (10-10)/5
        np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-7)   # (20-20)/10
        np.testing.assert_allclose(out[1, 0], -1.0, atol=1e-7)  # (5-10)/5
        np.testing.assert_allclose(out[1, 1], 1.0, atol=1e-7)   # (30-20)/10

    def test_shape_2d(self):
        data = np.ones((3, 2))
        stats = {'x': [0, 0, 0.0, 1.0], 'y': [0, 0, 0.0, 1.0]}
        out = normalize_array(data, stats, ['x', 'y'])
        assert out.shape == (3, 2)

    def test_shape_3d(self):
        data = np.ones((10, 5, 2))
        stats = {'x': [0, 0, 0.0, 1.0], 'y': [0, 0, 0.0, 1.0]}
        out = normalize_array(data, stats, ['x', 'y'])
        assert out.shape == (10, 5, 2)

    def test_dtype_float32(self):
        data = np.array([[1.0, 2.0]], dtype=np.float64)
        stats = {'a': [0, 0, 0.0, 1.0], 'b': [0, 0, 0.0, 1.0]}
        out = normalize_array(data, stats, ['a', 'b'])
        assert out.dtype == np.float32


# ===================================================================
# TestMmDayToM3s
# ===================================================================
class TestMmDayToM3s:
    """Convert mm/day -> m3/s using local catchment area."""

    def test_known_conversion(self):
        result = mm_day_to_m3s(np.array([[1.0]]), np.array([1.0]))
        expected = 1000.0 / 86400.0  # 0.011574074...
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-5)

    def test_nan_clamped(self):
        result = mm_day_to_m3s(np.array([[np.nan]]), np.array([1.0]))
        assert result[0, 0] == pytest.approx(1e-6)

    def test_negative_clamped(self):
        result = mm_day_to_m3s(np.array([[-5.0]]), np.array([1.0]))
        assert result[0, 0] == pytest.approx(1e-6)

    def test_zero_clamped(self):
        result = mm_day_to_m3s(np.array([[0.0]]), np.array([1.0]))
        assert result[0, 0] == pytest.approx(1e-6)

    def test_multi_basin(self):
        q = np.array([[1.0, 2.0], [1.0, 2.0]])
        areas = np.array([10.0, 100.0])
        result = mm_day_to_m3s(q, areas)
        conv_10 = 10.0 * 1000.0 / 86400.0
        conv_100 = 100.0 * 1000.0 / 86400.0
        np.testing.assert_allclose(result[0, 0], 1.0 * conv_10, rtol=1e-5)
        np.testing.assert_allclose(result[0, 1], 2.0 * conv_10, rtol=1e-5)
        np.testing.assert_allclose(result[1, 0], 1.0 * conv_100, rtol=1e-5)
        np.testing.assert_allclose(result[1, 1], 2.0 * conv_100, rtol=1e-5)

    def test_dtype_float32(self):
        result = mm_day_to_m3s(np.array([[5.0]]), np.array([50.0]))
        assert result.dtype == np.float32

    def test_all_values_ge_floor(self):
        np.random.seed(42)
        q = np.random.randn(20, 100).astype(np.float32)  # some negative
        areas = np.random.uniform(1, 1000, 20).astype(np.float32)
        result = mm_day_to_m3s(q, areas)
        assert (result >= 1e-6).all()

    def test_shape_preservation(self):
        q = np.ones((5, 10), dtype=np.float32)
        areas = np.ones(5, dtype=np.float32)
        result = mm_day_to_m3s(q, areas)
        assert result.shape == (5, 10)
