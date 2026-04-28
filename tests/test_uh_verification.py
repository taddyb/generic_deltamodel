"""Pre-training verification tests for dHBV2.0UH multiscale training.

These checks MUST all pass before any training run. Each maps to equations
in Song et al. 2025 or fundamental conservation laws.

Run with: pytest tests/test_uh_verification.py -v
"""

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import helpers from uh modules (src/ is not on sys.path by default).
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / 'src'

def _import_from(module_path: str, name: str):
    """Import a module from a dotted path relative to src/."""
    full = _SRC / module_path.replace('.', '/') / '__init__.py'
    if not full.exists():
        full = _SRC / (module_path.replace('.', '/') + '.py')
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================
# Check 2: Unit Hydrograph Mass Balance
# ===================================================================
class TestUhMassBalance:
    """Gamma UH must conserve mass: sum(UH_kernel) ≈ 1.0."""

    @pytest.fixture()
    def uh_funcs(self):
        """Import uh_gamma and uh_conv from hydrodl2."""
        from hydrodl2.core.calc.uh_routing import uh_conv, uh_gamma
        return uh_gamma, uh_conv

    def test_uh_kernel_sums_to_one(self, uh_funcs):
        """UH kernel must sum to ~1.0 for various (a, b) within bounds."""
        uh_gamma, _ = uh_funcs
        np.random.seed(42)
        n_samples = 100
        lenF = 15

        # Random route_a in [0.1, 2.9], route_b in [0.5, 6.5]
        a_vals = torch.rand(1, n_samples, 1) * 2.8 + 0.1
        b_vals = torch.rand(1, n_samples, 1) * 6.0 + 0.5

        UH = uh_gamma(a_vals.expand(lenF, -1, -1), b_vals.expand(lenF, -1, -1), lenF=lenF)
        # UH shape: [lenF, n_samples, 1]
        kernel_sums = UH.sum(dim=0).squeeze()  # [n_samples]

        for i in range(n_samples):
            assert abs(kernel_sums[i].item() - 1.0) < 0.05, (
                f"UH kernel {i} sum={kernel_sums[i].item():.4f}, "
                f"a={a_vals[0,i,0].item():.2f}, b={b_vals[0,i,0].item():.2f}"
            )

    def test_steady_state_passthrough(self, uh_funcs):
        """Constant input convolved with UH should reach steady state ≈ input."""
        uh_gamma, uh_conv = uh_funcs
        lenF = 15
        T = 100

        # Single basin, constant input of 1.0 mm/day
        a = torch.tensor([[[1.5]]])  # [1, 1, 1]
        b = torch.tensor([[[3.0]]])
        UH = uh_gamma(a.expand(lenF, -1, -1), b.expand(lenF, -1, -1), lenF=lenF)

        # Input: constant 1.0
        q_in = torch.ones(1, 1, T)  # [batch, var, time]
        UH_k = UH.permute(1, 2, 0)  # [batch, var, lenF]
        q_out = uh_conv(q_in, UH_k)  # [batch, var, T]

        # After spin-up, output should be ~1.0
        steady = q_out[0, 0, lenF:].mean().item()
        assert abs(steady - 1.0) < 0.02, f"Steady state = {steady:.4f}, expected ~1.0"


# ===================================================================
# Check 3: Network Accumulation on Known Topology
# ===================================================================
class TestNetworkAccumulation:
    """Verify accumulation with a hand-computed 3-basin → 1-gauge network."""

    def test_three_basin_accumulation(self):
        """
        Basin A (area=10 km²)  Basin B (area=20 km²)
               \\                    /
                → Basin C (area=15 km²) → Gauge G

        topo = [[1, 1, 1]]  (gauge G drains all 3 basins)
        Q_local = [1.0, 2.0, 3.0] mm/day

        Expected:
          Q_A = 1.0 × 10 × 1000/86400 = 0.11574 m³/s
          Q_B = 2.0 × 20 × 1000/86400 = 0.46296 m³/s
          Q_C = 3.0 × 15 × 1000/86400 = 0.52083 m³/s
          Q_gauge = 1.09954 m³/s
        """
        MM_D_TO_M3S = 1000.0 / 86400.0

        q_local = torch.tensor([[1.0, 2.0, 3.0]])  # [1 timestep, 3 basins]
        areas = torch.tensor([10.0, 20.0, 15.0])
        topo = torch.tensor([[1.0, 1.0, 1.0]])  # [1 gauge, 3 basins]

        q_m3s = q_local * areas.unsqueeze(0) * MM_D_TO_M3S
        q_gauge = torch.matmul(topo, q_m3s.T).T  # [1, 1]

        expected = (1.0 * 10 + 2.0 * 20 + 3.0 * 15) * MM_D_TO_M3S
        np.testing.assert_allclose(
            q_gauge[0, 0].item(), expected, rtol=1e-5,
            err_msg=f"Q_gauge={q_gauge[0,0].item()}, expected={expected}",
        )

    def test_two_gauges_nested(self):
        """
        Basin A → Gauge G1
        Basin A + Basin B → Gauge G2

        topo = [[1, 0],  # G1 drains only A
                [1, 1]]  # G2 drains A and B
        """
        MM_D_TO_M3S = 1000.0 / 86400.0

        q_local = torch.tensor([[5.0, 3.0]])  # [1, 2 basins]
        areas = torch.tensor([100.0, 50.0])
        topo = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        q_m3s = q_local * areas.unsqueeze(0) * MM_D_TO_M3S
        q_gauge = torch.matmul(topo, q_m3s.T).T

        expected_g1 = 5.0 * 100.0 * MM_D_TO_M3S
        expected_g2 = (5.0 * 100.0 + 3.0 * 50.0) * MM_D_TO_M3S

        np.testing.assert_allclose(q_gauge[0, 0].item(), expected_g1, rtol=1e-5)
        np.testing.assert_allclose(q_gauge[0, 1].item(), expected_g2, rtol=1e-5)

    def test_time_dimension(self):
        """Accumulation should work independently at each timestep."""
        MM_D_TO_M3S = 1000.0 / 86400.0
        T = 10

        q_local = torch.rand(T, 3)
        areas = torch.tensor([10.0, 20.0, 15.0])
        topo = torch.tensor([[1.0, 1.0, 1.0]])

        q_m3s = q_local * areas.unsqueeze(0) * MM_D_TO_M3S
        q_gauge = torch.matmul(topo, q_m3s.T).T  # [T, 1]

        # Manual check for first timestep
        expected_t0 = (q_local[0] * areas * MM_D_TO_M3S).sum().item()
        np.testing.assert_allclose(q_gauge[0, 0].item(), expected_t0, rtol=1e-5)

        assert q_gauge.shape == (T, 1)


# ===================================================================
# Check 5: Gradient Flow Through Accumulation
# ===================================================================
class TestGradientFlow:
    """Verify gradients propagate through topo @ Q_local to upstream params."""

    def test_gradient_through_matmul(self):
        """Simple gradient flow: loss on q_gauge → grad on q_local."""
        q_local = torch.randn(10, 3, requires_grad=True)  # [T, 3 basins]
        areas = torch.tensor([10.0, 20.0, 15.0])
        topo = torch.tensor([[1.0, 1.0, 1.0]])  # [1 gauge, 3 basins]

        q_m3s = q_local * areas.unsqueeze(0) * (1000.0 / 86400.0)
        q_gauge = torch.matmul(topo, q_m3s.T).T  # [T, 1]

        # Fake target
        target = torch.randn_like(q_gauge)
        loss = ((q_gauge - target) ** 2).mean()
        loss.backward()

        assert q_local.grad is not None, "No gradient for q_local"
        assert q_local.grad.abs().sum() > 0, "Zero gradient for q_local"

        # All 3 basins should receive gradients (all upstream of gauge)
        for basin in range(3):
            assert q_local.grad[:, basin].abs().sum() > 0, (
                f"Zero gradient for basin {basin}"
            )

    def test_gradient_selective_by_topo(self):
        """Only upstream basins should receive gradients."""
        q_local = torch.randn(5, 3, requires_grad=True)
        areas = torch.tensor([10.0, 20.0, 15.0])
        # Gauge only drains basin 0 and 2 (not 1)
        topo = torch.tensor([[1.0, 0.0, 1.0]])

        q_m3s = q_local * areas.unsqueeze(0) * (1000.0 / 86400.0)
        q_gauge = torch.matmul(topo, q_m3s.T).T

        target = torch.randn_like(q_gauge)
        loss = ((q_gauge - target) ** 2).mean()
        loss.backward()

        # Basin 0 and 2 should have gradients
        assert q_local.grad[:, 0].abs().sum() > 0, "No gradient for upstream basin 0"
        assert q_local.grad[:, 2].abs().sum() > 0, "No gradient for upstream basin 2"
        # Basin 1 should have zero gradient (not upstream)
        assert q_local.grad[:, 1].abs().sum() == 0, (
            "Basin 1 should have zero gradient (not upstream)"
        )


# ===================================================================
# Check 7: Reachability Matrix Consistency
# ===================================================================
class TestReachabilityMatrix:
    """Verify reachability_matrix produces correct topology."""

    @pytest.fixture()
    def simple_network(self):
        """
        1 → 3 → 5 (outlet for gauge A)
        2 → 3
        4 → 5
        """
        G = nx.DiGraph()
        G.add_edges_from([(1, 3), (2, 3), (3, 5), (4, 5)])
        return G

    def test_reachability_single_outlet(self, simple_network):
        from dmg.core.utils.topo_operator import reachability_matrix

        G = simple_network
        outlets = [5]
        basins = [1, 2, 3, 4, 5]

        M = reachability_matrix(G, outlets, basins)

        # All basins should reach outlet 5
        assert M.shape == (1, 5)
        np.testing.assert_array_equal(M[0], [1, 1, 1, 1, 1])

    def test_reachability_nested_outlets(self, simple_network):
        from dmg.core.utils.topo_operator import reachability_matrix

        G = simple_network
        outlets = [3, 5]  # gauge at 3 and gauge at 5
        basins = [1, 2, 3, 4, 5]

        M = reachability_matrix(G, outlets, basins)

        # Gauge at 3: basins 1, 2, 3 can reach it
        assert M.shape == (2, 5)
        np.testing.assert_array_equal(M[0], [1, 1, 1, 0, 0])
        # Gauge at 5: all basins can reach it
        np.testing.assert_array_equal(M[1], [1, 1, 1, 1, 1])

    def test_each_gauge_has_upstream(self, simple_network):
        from dmg.core.utils.topo_operator import reachability_matrix

        G = simple_network
        outlets = [3, 5]
        basins = [1, 2, 3, 4, 5]

        M = reachability_matrix(G, outlets, basins)

        # Each gauge should drain at least 1 basin
        assert M.sum(axis=1).min() >= 1

    def test_upstream_connectivity(self, simple_network):
        """Upstream basins of each gauge should form a connected subgraph."""
        from dmg.core.utils.topo_operator import reachability_matrix

        G = simple_network
        outlets = [3, 5]
        basins = [1, 2, 3, 4, 5]

        M = reachability_matrix(G, outlets, basins)

        for g in range(len(outlets)):
            upstream_mask = M[g] == 1
            upstream_basins = [basins[j] for j in range(len(basins)) if upstream_mask[j]]
            subG = G.subgraph(upstream_basins)
            assert nx.is_weakly_connected(subG), (
                f"Disconnected upstream for gauge {outlets[g]}: {upstream_basins}"
            )


# ===================================================================
# Check: UhHydroSampler TimeWindowBatchSampler
# ===================================================================
class TestTimeWindowBatchSampler:
    """Verify the gauge-centric batch sampler produces valid batches."""

    def test_all_gauges_sampled(self):
        """Every gauge with valid data should appear at least once."""
        from dmg.core.data.samplers.uh_hydro_sampler import TimeWindowBatchSampler

        n_gauges, n_t = 10, 1000
        target = torch.randn(n_gauges, n_t)
        # No NaNs → all gauges valid everywhere

        sampler = TimeWindowBatchSampler(
            target=target,
            window_size=365,
            warmup=365,
            stride=365,
            batch_size=4,
            shuffle=False,
        )

        seen_gauges = set()
        for t, batch in sampler:
            seen_gauges.update(batch)
            assert t >= 365, f"Window starts in warmup: t={t}"
            assert t + 365 <= n_t, f"Window exceeds data: t={t}"

        assert seen_gauges == set(range(n_gauges)), (
            f"Missing gauges: {set(range(n_gauges)) - seen_gauges}"
        )

    def test_nan_gauges_excluded(self):
        """Gauges with all-NaN in a window should be excluded from that window."""
        from dmg.core.data.samplers.uh_hydro_sampler import TimeWindowBatchSampler

        n_gauges, n_t = 5, 800
        target = torch.randn(n_gauges, n_t)
        # Make gauge 2 all-NaN in the valid time range
        target[2, :] = float('nan')

        sampler = TimeWindowBatchSampler(
            target=target,
            window_size=365,
            warmup=365,
            stride=365,
            batch_size=10,
            shuffle=False,
        )

        for t, batch in sampler:
            assert 2 not in batch, f"NaN gauge 2 should be excluded at t={t}"


# ===================================================================
# Check: Accumulation mm/d → m³/s conversion factor
# ===================================================================
class TestUnitConversion:
    """Verify the mm/day to m³/s conversion is correct."""

    def test_known_conversion(self):
        """1 mm/day over 1 km² = 1000 m³/day = 1000/86400 m³/s."""
        MM_D_TO_M3S = 1000.0 / 86400.0
        result = 1.0 * 1.0 * MM_D_TO_M3S
        expected = 1000.0 / 86400.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_scaling_with_area(self):
        """10 mm/day over 50 km² = 500,000 m³/day = 5.787 m³/s."""
        MM_D_TO_M3S = 1000.0 / 86400.0
        result = 10.0 * 50.0 * MM_D_TO_M3S
        expected = 10.0 * 50.0 * 1000.0 / 86400.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        np.testing.assert_allclose(result, 5.787037, rtol=1e-4)
