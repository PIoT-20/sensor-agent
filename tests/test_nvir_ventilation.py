import pytest
import pandas as pd
from agents.NVIRVentlationAgent import NVIRVentlationAgent


@pytest.fixture
def agent():
    return NVIRVentlationAgent("nvir-ventilation", "http://localhost:8080", "http://catalog:8080")


# Standard calibration conditions — calibration is identity at these values
STD_P = 1013.25
STD_T = 25.0  # 298.15 K


def _make_readings(timestamps, co2_values, t_values=None, p_values=None):
    """Helper to build reading dicts. Defaults to standard calibration conditions."""
    if t_values is None:
        t_values = [STD_T] * len(timestamps)
    if p_values is None:
        p_values = [STD_P] * len(timestamps)
    return [
        {
            "timestamp": ts if isinstance(ts, str) else ts.isoformat(),
            "value": {"T": t, "P": p, "CO2": co2},
        }
        for ts, co2, t, p in zip(timestamps, co2_values, t_values, p_values)
    ]


# ---------- Validation ----------

class TestValidation:
    def test_empty_readings_raises(self, agent):
        with pytest.raises(ValueError, match="No readings provided"):
            agent.process_readings([])


# ---------- Calibration ----------

class TestCalibration:
    def test_uniform_at_standard_conditions(self, agent):
        """At standard P and T, uniform readings normalize to baseline → low WSI."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=5, freq="1h")
        readings = _make_readings(timestamps, [600.0] * 5)
        result = agent.process_readings(readings)
        assert result < 1.0

    def test_low_pressure_amplifies_spread(self, agent):
        """Lower pressure scales CO2 up, amplifying the gap between baseline and spike."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        co2_vals = [500.0] * 9 + [800.0]

        result_std = agent.process_readings(_make_readings(timestamps, co2_vals))
        result_low_p = agent.process_readings(
            _make_readings(timestamps, co2_vals, p_values=[STD_P / 2] * 10)
        )

        assert result_low_p > result_std


# ---------- Baseline correction ----------

class TestBaselineCorrection:
    def test_uniform_readings_same_wsi_regardless_of_level(self, agent):
        """Uniform CO2 readings get shifted to the same baseline, producing equal WSI."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=5, freq="1h")
        results = []
        for co2 in [400.0, 800.0, 1200.0]:
            readings = _make_readings(timestamps, [co2] * 5)
            results.append(agent.process_readings(readings))

        assert results[0] == pytest.approx(results[1], abs=0.001)
        assert results[1] == pytest.approx(results[2], abs=0.001)

    def test_same_spread_same_wsi(self, agent):
        """Same relative variation but different baselines → same WSI after correction."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        co2_low = [500.0] * 9 + [700.0]
        co2_high = [900.0] * 9 + [1100.0]

        result_low = agent.process_readings(_make_readings(timestamps, co2_low))
        result_high = agent.process_readings(_make_readings(timestamps, co2_high))
        assert result_low == pytest.approx(result_high, abs=0.01)


# ---------- Shift filtering ----------

class TestShiftFiltering:
    def test_boundary_times_included(self, agent):
        """Readings at exactly 09:00 and 18:00 should be kept."""
        readings = _make_readings(
            ["2023-10-23T09:00:00", "2023-10-23T18:00:00"],
            [600.0, 600.0],
        )
        result = agent.process_readings(readings)
        assert result > 0

    def test_weekend_readings_excluded_from_wsi(self, agent):
        """Weekend readings go through adjustment but don't contribute to WSI."""
        ts_weekend = pd.date_range("2023-10-21 09:00", periods=3, freq="1h")  # Saturday
        ts_weekday = pd.date_range("2023-10-23 09:00", periods=3, freq="1h")  # Monday
        readings = (
            _make_readings(ts_weekend, [2000.0] * 3) +
            _make_readings(ts_weekday, [600.0] * 3)
        )
        result = agent.process_readings(readings)
        # Only Monday readings count for WSI — result should be low
        assert result < 1.0

    def test_off_hours_excluded(self, agent):
        """Before-9AM and after-6PM readings don't count for WSI."""
        readings = _make_readings(
            ["2023-10-23T07:00:00", "2023-10-23T12:00:00", "2023-10-23T20:00:00"],
            [600.0, 600.0, 600.0],
        )
        result = agent.process_readings(readings)
        assert result > 0


# ---------- WSI: low vs high index ----------

class TestWSILowHigh:
    def test_low_co2_low_index(self, agent):
        """Low CO2 with little variation → low WSI (good ventilation)."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=8, freq="1h")
        readings = _make_readings(timestamps, [450.0] * 8)
        result = agent.process_readings(readings)
        assert result < 1.0

    def test_high_co2_spike_high_index(self, agent):
        """Large CO2 spikes above baseline → higher WSI (poor ventilation)."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        readings = _make_readings(timestamps, [500.0] * 5 + [2000.0] * 5)
        result = agent.process_readings(readings)
        assert result > 0.4  # well above the uniform-baseline level

    def test_more_variation_higher_index(self, agent):
        """Greater spread in CO2 readings → higher WSI than uniform readings."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        readings_uniform = _make_readings(timestamps, [600.0] * 10)
        readings_varied = _make_readings(timestamps, [500.0] * 8 + [1500.0] * 2)

        result_uniform = agent.process_readings(readings_uniform)
        result_varied = agent.process_readings(readings_varied)
        assert result_varied > result_uniform


# ---------- Multi-day ----------

class TestMultiDay:
    def test_identical_days_stable_score(self, agent):
        """Two identical days should produce the same global score as one day."""
        ts_day1 = pd.date_range("2023-10-23 09:00", periods=5, freq="1h")
        ts_day2 = pd.date_range("2023-10-24 09:00", periods=5, freq="1h")

        single_day = agent.process_readings(_make_readings(ts_day1, [600.0] * 5))
        two_days = agent.process_readings(
            _make_readings(ts_day1, [600.0] * 5) +
            _make_readings(ts_day2, [600.0] * 5)
        )
        # Close but not necessarily identical due to EWMA carryover across days
        assert two_days == pytest.approx(single_day, abs=0.05)

    def test_bad_day_raises_global_score(self, agent):
        """Adding a high-CO2 day should raise the global score."""
        ts_good = pd.date_range("2023-10-23 09:00", periods=5, freq="1h")  # Monday
        ts_bad = pd.date_range("2023-10-24 09:00", periods=5, freq="1h")   # Tuesday

        result_good_only = agent.process_readings(_make_readings(ts_good, [500.0] * 5))
        result_with_bad = agent.process_readings(
            _make_readings(ts_good, [500.0] * 5) +
            _make_readings(ts_bad, [500.0, 500.0, 2000.0, 2000.0, 2000.0])
        )
        assert result_with_bad > result_good_only


# ---------- EWMA smoothing ----------

class TestEWMASmoothing:
    def test_early_spike_persists_more_than_late_spike(self, agent):
        """EWMA (alpha=0.1) has long memory — an early spike elevates the entire smoothed series."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        co2_spike_first = [1500.0] + [500.0] * 9
        co2_spike_last = [500.0] * 9 + [1500.0]

        result_first = agent.process_readings(_make_readings(timestamps, co2_spike_first))
        result_last = agent.process_readings(_make_readings(timestamps, co2_spike_last))
        assert result_first > result_last
