import pytest
import pandas as pd
from agents.EUOSHANoiseAgent import EUOSHANoiseAgent

@pytest.fixture
def agent():
    return EUOSHANoiseAgent()

def test_process_readings_basic(agent): 
    timestamps = pd.date_range(start="2023-10-23 09:00:00", end="2023-10-23 18:00:00", freq="1h")
    readings = []
    for ts in timestamps:
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"dB": 85.0}
        })
        
    result = agent.process_readings(readings)
    
    assert result == pytest.approx(1.0, 0.01)

def test_process_readings_empty(agent):
    with pytest.raises(ValueError, match="No readings provided"):
        agent.process_readings([])

def test_process_readings_outside_shift(agent):
    readings = [
        {
            "timestamp": "2023-10-22T12:00:00", # Sunday
            "value": {"dB": 100.0}
        }
    ]
    
    with pytest.raises(ValueError, match="No readings during work shift"):
        agent.process_readings(readings)

def test_process_readings_low_exposure(agent):
    # Readings varying but generally low (below 85dB)
    # 60dB, 70dB, 65dB, 75dB
    timestamps = pd.date_range(start="2023-10-23 09:00:00", periods=4, freq="1h")
    values = [60.0, 70.0, 65.0, 75.0]
    readings = []
    for ts, val in zip(timestamps, values):
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"dB": val}
        })
        
    result = agent.process_readings(readings)
    
    # The average energy will correspond to a dB value < 85
    assert result < 1.0

def test_process_readings_high_exposure(agent):
    # Readings varying, some high
    # 80dB, 90dB, 85dB, 100dB
    # 100dB dominates significantly (energy is logarithmic)
    timestamps = pd.date_range(start="2023-10-23 09:00:00", periods=4, freq="1h")
    values = [80.0, 90.0, 85.0, 100.0]
    readings = []
    for ts, val in zip(timestamps, values):
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"dB": val}
        })
        
    result = agent.process_readings(readings)
    
    # The average energy will correspond to a dB value > 85
    assert result > 1.0

def test_process_readings_calculation_accuracy(agent):
    # Specific calculation test to verify correctness
    # Day 1: 80 dB, 90 dB
    # Day 2: 85 dB, 85 dB
    
    # Day 1 (Mon)
    ts1 = ["2023-10-23 10:00:00", "2023-10-23 11:00:00"]
    val1 = [80.0, 90.0]
    
    # Day 2 (Tue)
    ts2 = ["2023-10-24 10:00:00", "2023-10-24 11:00:00"]
    val2 = [85.0, 85.0]
    
    readings = []
    for t, v in zip(ts1 + ts2, val1 + val2):
        readings.append({
            "timestamp": t,
            "value": {"dB": v}
        })
        
    # Calculation:
    # Day 1 Energy Mean: (10^8 + 10^9) / 2 = 5.5 * 10^8
    # Day 2 Energy Mean: (10^8.5 + 10^8.5) / 2 = 10^8.5 ~= 3.16228 * 10^8
    # Total Mean Energy: (5.5 + 3.16228) / 2 * 10^8 = 4.33114 * 10^8
    # Leq = 10 * log10(4.33114 * 10^8) = 80 + 10 * log10(4.33114) 
    #     = 80 + 6.366 = 86.366 dB
    # Ratio = 86.366 / 85 ~= 1.01607
    
    result = agent.process_readings(readings)
    expected_leq = 86.366
    expected_ratio = expected_leq / 85.0
    
    assert result == pytest.approx(expected_ratio, 0.001)
