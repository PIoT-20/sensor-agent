import pytest
import numpy as np
import pandas as pd
from agents.TWANoiseAgent import TWANoiseAgent


@pytest.fixture
def agent():
    return TWANoiseAgent()


def _make_readings(timestamps, db_values):
    """Helper to build a list of reading dicts from timestamps and dB values."""
    return [
        {"timestamp": ts if isinstance(ts, str) else ts.isoformat(), "value": {"db": db}}
        for ts, db in zip(timestamps, db_values)
    ]


# ---------- Validation ----------

class TestValidation:
    def test_empty_readings_raises(self, agent):
        with pytest.raises(ValueError, match="No readings provided"):
            agent.process_readings([])

    def test_weekend_only_readings_raises(self, agent):
        readings = [{"timestamp": "2023-10-22T12:00:00", "value": {"db": 90.0}}]  # Sunday
        with pytest.raises(ValueError, match="No readings during work hours"):
            agent.process_readings(readings)

    def test_outside_shift_hours_raises(self, agent):
        readings = [
            {"timestamp": "2023-10-23T07:00:00", "value": {"db": 90.0}},  # Monday, before 9AM
            {"timestamp": "2023-10-23T20:00:00", "value": {"db": 90.0}},  # Monday, after 6PM
        ]
        with pytest.raises(ValueError, match="No readings during work hours"):
            agent.process_readings(readings)


# ---------- Shift filtering ----------

class TestShiftFiltering:
    def test_boundary_times_are_included(self, agent):
        """Readings at exactly 09:00 and 18:00 should be kept."""
        readings = _make_readings(
            ["2023-10-23T09:00:00", "2023-10-23T18:00:00"],
            [85.0, 85.0],
        )
        result = agent.process_readings(readings)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_non_work_readings_are_filtered_out(self, agent):
        """Only work-shift readings contribute to the result."""
        readings = [
            {"timestamp": "2023-10-23T08:00:00", "value": {"db": 120.0}},  # before shift — ignored
            {"timestamp": "2023-10-23T12:00:00", "value": {"db": 85.0}},   # during shift — kept
            {"timestamp": "2023-10-23T19:00:00", "value": {"db": 120.0}},  # after shift — ignored
        ]
        result = agent.process_readings(readings)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_weekday_filter(self, agent):
        """Saturday and Sunday readings are discarded; weekday readings are kept."""
        readings = [
            {"timestamp": "2023-10-21T12:00:00", "value": {"db": 120.0}},  # Saturday
            {"timestamp": "2023-10-22T12:00:00", "value": {"db": 120.0}},  # Sunday
            {"timestamp": "2023-10-23T12:00:00", "value": {"db": 85.0}},   # Monday
        ]
        result = agent.process_readings(readings)
        assert result == pytest.approx(1.0, abs=0.001)


# ---------- WSI computation ----------

class TestWSIComputation:
    def test_constant_at_threshold_returns_one(self, agent):
        """Uniform 85 dB readings over a full shift should yield WSI = 1.0."""
        timestamps = pd.date_range("2023-10-23 09:00", "2023-10-23 17:00", freq="1h")
        readings = _make_readings(timestamps, [85.0] * len(timestamps))
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)

    def test_below_threshold_returns_less_than_one(self, agent):
        timestamps = pd.date_range("2023-10-23 09:00", periods=4, freq="1h")
        readings = _make_readings(timestamps, [60.0, 70.0, 65.0, 75.0])
        assert agent.process_readings(readings) < 1.0

    def test_above_threshold_returns_greater_than_one(self, agent):
        timestamps = pd.date_range("2023-10-23 09:00", periods=4, freq="1h")
        readings = _make_readings(timestamps, [80.0, 90.0, 85.0, 100.0])
        assert agent.process_readings(readings) > 1.0

    def test_single_reading(self, agent):
        """A single 90 dB reading should yield WSI = 90/85."""
        readings = [{"timestamp": "2023-10-23T12:00:00", "value": {"db": 90.0}}]
        assert agent.process_readings(readings) == pytest.approx(90.0 / 85.0, abs=0.001)

    def test_identical_readings_same_as_single(self, agent):
        """Repeated identical readings should produce the same WSI as a single reading."""
        timestamps = pd.date_range("2023-10-23 09:00", periods=5, freq="1h")
        readings = _make_readings(timestamps, [90.0] * 5)
        assert agent.process_readings(readings) == pytest.approx(90.0 / 85.0, abs=0.001)


# ---------- Multi-day ----------

class TestMultiDay:
    def test_multi_day_averages_daily_wsi(self, agent):
        """
        Day 1 (Mon): 80 dB, 90 dB
          energy mean = (10^8 + 10^9) / 2 = 5.5e8
          NEL = 10*log10(5.5e8) ≈ 87.404 dB  →  WSI ≈ 1.02829
        Day 2 (Tue): 85 dB, 85 dB
          energy mean = 10^8.5
          NEL = 85 dB  →  WSI = 1.0
        Global = mean(1.02829, 1.0) ≈ 1.01414
        """
        readings = _make_readings(
            ["2023-10-23 10:00:00", "2023-10-23 11:00:00",
             "2023-10-24 10:00:00", "2023-10-24 11:00:00"],
            [80.0, 90.0, 85.0, 85.0],
        )
        # Compute expected value from first principles
        day1_energy_mean = (10 ** 8 + 10 ** 9) / 2
        day1_nel = 10 * np.log10(day1_energy_mean)
        day2_nel = 85.0
        expected = ((day1_nel / 85.0) + (day2_nel / 85.0)) / 2

        result = agent.process_readings(readings)
        assert result == pytest.approx(expected, abs=0.001)

    def test_days_weighted_equally(self, agent):
        """Each day contributes equally regardless of how many readings it has."""
        readings = _make_readings(
            # Monday: 6 readings at 85 dB
            list(pd.date_range("2023-10-23 09:00", periods=6, freq="1h")) +
            # Tuesday: 1 reading at 85 dB
            ["2023-10-24T12:00:00"],
            [85.0] * 6 + [85.0],
        )
        assert agent.process_readings(readings) == pytest.approx(1.0, abs=0.001)