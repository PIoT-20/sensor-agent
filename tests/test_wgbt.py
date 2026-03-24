import pytest
import pandas as pd
from agents.WBGTAgent import WBGTAgent


@pytest.fixture
def agent():
    return WBGTAgent()


def _make_readings(timestamps, tnw_values, tg_values):
    """Helper to build a list of reading dicts from timestamps and sensor values."""
    return [
        {"timestamp": ts if isinstance(ts, str) else ts.isoformat(), "value": {"Tnw": tnw, "Tg": tg}}
        for ts, tnw, tg in zip(timestamps, tnw_values, tg_values)
    ]


# ---------- Validation ----------

class TestValidation:
    def test_empty_readings_raises(self, agent):
        with pytest.raises(ValueError, match="No readings provided"):
            agent.process_readings([])

    def test_weekend_only_readings_raises(self, agent):
        readings = [{"timestamp": "2023-10-22T12:00:00", "value": {"Tnw": 10.0, "Tg": 10.0}}]  # Sunday
        with pytest.raises(ValueError, match="No readings during work shift"):
            agent.process_readings(readings)

    def test_outside_shift_hours_raises(self, agent):
        readings = [
            {"timestamp": "2023-10-23T07:00:00", "value": {"Tnw": 30.0, "Tg": 30.0}},  # before 9AM
            {"timestamp": "2023-10-23T20:00:00", "value": {"Tnw": 30.0, "Tg": 30.0}},  # after 6PM
        ]
        with pytest.raises(ValueError, match="No readings during work shift"):
            agent.process_readings(readings)


# ---------- Shift filtering ----------

class TestShiftFiltering:
    def test_boundary_times_are_included(self, agent):
        """Readings at exactly 09:00 and 18:00 should be kept."""
        readings = _make_readings(
            ["2023-10-23T09:00:00", "2023-10-23T18:00:00"],
            [25.0, 25.0], [25.0, 25.0],
        )
        result = agent.process_readings(readings)
        # WBGT = 0.7*25 + 0.3*25 = 25, WSI = 25/25 = 1.0
        assert result == pytest.approx(1.0, abs=0.001)

    def test_non_work_readings_are_filtered_out(self, agent):
        """Only work-shift readings contribute to the result."""
        readings = [
            {"timestamp": "2023-10-23T08:00:00", "value": {"Tnw": 50.0, "Tg": 50.0}},  # before shift
            {"timestamp": "2023-10-23T12:00:00", "value": {"Tnw": 25.0, "Tg": 25.0}},  # during shift
            {"timestamp": "2023-10-23T19:00:00", "value": {"Tnw": 50.0, "Tg": 50.0}},  # after shift
        ]
        result = agent.process_readings(readings)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_weekday_filter(self, agent):
        """Saturday and Sunday readings are discarded; weekday readings are kept."""
        readings = [
            {"timestamp": "2023-10-21T12:00:00", "value": {"Tnw": 50.0, "Tg": 50.0}},  # Saturday
            {"timestamp": "2023-10-22T12:00:00", "value": {"Tnw": 50.0, "Tg": 50.0}},  # Sunday
            {"timestamp": "2023-10-23T12:00:00", "value": {"Tnw": 25.0, "Tg": 25.0}},  # Monday
        ]
        result = agent.process_readings(readings)
        assert result == pytest.approx(1.0, abs=0.001)


# ---------- WBGT formula ----------

class TestWBGTFormula:
    def test_wbgt_weighted_combination(self, agent):
        """WBGT = 0.7*Tnw + 0.3*Tg — verify weighting is correct."""
        # Tnw=20, Tg=40 → WBGT = 0.7*20 + 0.3*40 = 14 + 12 = 26
        # WSI = 26/25 = 1.04
        readings = [{"timestamp": "2023-10-23T12:00:00", "value": {"Tnw": 20.0, "Tg": 40.0}}]
        result = agent.process_readings(readings)
        assert result == pytest.approx(26.0 / 25.0, abs=0.001)

    def test_equal_sensors_yield_same_wbgt(self, agent):
        """When Tnw == Tg, WBGT equals that temperature."""
        readings = [{"timestamp": "2023-10-23T12:00:00", "value": {"Tnw": 30.0, "Tg": 30.0}}]
        result = agent.process_readings(readings)
        # WBGT = 30, WSI = 30/25 = 1.2
        assert result == pytest.approx(30.0 / 25.0, abs=0.001)


# ---------- WSI computation ----------

class TestWSIComputation:
    def test_at_threshold_returns_one(self, agent):
        """Uniform WBGT = 25°C readings should yield WSI = 1.0."""
        timestamps = pd.date_range("2023-10-23 09:00", "2023-10-23 17:00", freq="1h")
        readings = _make_readings(timestamps, [25.0] * len(timestamps), [25.0] * len(timestamps))
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)

    def test_below_threshold(self, agent):
        """Low temperature readings → WSI < 1.0."""
        timestamps = pd.date_range("2023-10-23 09:00", "2023-10-23 17:00", freq="1h")
        readings = _make_readings(timestamps, [10.0] * len(timestamps), [10.0] * len(timestamps))
        # WBGT = 10, WSI = 10/25 = 0.4
        assert agent.process_readings(readings) == pytest.approx(0.4, abs=0.001)

    def test_above_threshold(self, agent):
        """High temperature readings → WSI > 1.0."""
        timestamps = pd.date_range("2023-10-23 09:00", "2023-10-23 17:00", freq="1h")
        readings = _make_readings(timestamps, [30.0] * len(timestamps), [30.0] * len(timestamps))
        # WBGT = 30, WSI = 30/25 = 1.2
        assert agent.process_readings(readings) == pytest.approx(1.2, abs=0.001)

    def test_single_reading(self, agent):
        """A single reading: avg == p95, so WSI = WBGT / 25."""
        readings = [{"timestamp": "2023-10-23T12:00:00", "value": {"Tnw": 20.0, "Tg": 20.0}}]
        assert agent.process_readings(readings) == pytest.approx(20.0 / 25.0, abs=0.001)


# ---------- Weighted aggregation (ALPHA) ----------

class TestWeightedAggregation:
    def test_alpha_weighting_with_outlier(self, agent):
        """
        With mixed readings, the 95th percentile pulls the WSI above the simple average.
        ALPHA=0.4 weights the average, (1-ALPHA)=0.6 weights the 95th percentile.
        """
        # 19 readings at WBGT=20, 1 reading at WBGT=30
        timestamps = pd.date_range("2023-10-23 09:00", periods=20, freq="15min")
        tnw_vals = [20.0] * 19 + [30.0]
        tg_vals = [20.0] * 19 + [30.0]
        readings = _make_readings(timestamps, tnw_vals, tg_vals)

        wbgt_values = pd.Series([20.0] * 19 + [30.0])
        avg = wbgt_values.mean() / 25.0
        p95 = wbgt_values.quantile(0.95) / 25.0
        expected = 0.4 * avg + 0.6 * p95

        result = agent.process_readings(readings)
        assert result == pytest.approx(expected, abs=0.001)

    def test_uniform_readings_alpha_irrelevant(self, agent):
        """When all readings are identical, avg == p95 so ALPHA has no effect."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=10, freq="30min")
        readings = _make_readings(timestamps, [25.0] * 10, [25.0] * 10)
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)


# ---------- Multi-day ----------

class TestMultiDay:
    def test_multi_day_averages_daily_wsi(self, agent):
        """
        Day 1 (Mon): all WBGT=20 → WSI = 20/25 = 0.8
        Day 2 (Tue): all WBGT=30 → WSI = 30/25 = 1.2
        Global = mean(0.8, 1.2) = 1.0
        """
        ts_day1 = pd.date_range("2023-10-23 09:00", periods=4, freq="1h")
        ts_day2 = pd.date_range("2023-10-24 09:00", periods=4, freq="1h")
        readings = (
            _make_readings(ts_day1, [20.0] * 4, [20.0] * 4) +
            _make_readings(ts_day2, [30.0] * 4, [30.0] * 4)
        )
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)

    def test_days_weighted_equally(self, agent):
        """Each day contributes equally regardless of reading count."""
        readings = (
            # Monday: 6 readings at WBGT=25
            _make_readings(
                pd.date_range("2023-10-23 09:00", periods=6, freq="1h"),
                [25.0] * 6, [25.0] * 6,
            ) +
            # Tuesday: 1 reading at WBGT=25
            _make_readings(["2023-10-24T12:00:00"], [25.0], [25.0])
        )
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)
