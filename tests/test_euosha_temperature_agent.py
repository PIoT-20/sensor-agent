import pytest
import pandas as pd
from agents.EUOSHATemperatureAgent import EUOSHATemperatureAgent

@pytest.fixture
def agent():
    return EUOSHATemperatureAgent()

def test_process_readings_empty(agent):
    with pytest.raises(ValueError, match="No readings provided"):
        agent.process_readings([])

def test_process_readings_outside_shift(agent):
    readings = [
        {
            "timestamp": "2023-10-22T12:00:00", # Sunday
            "value": {"Tnw": 10.0, "Tg": 10.0}
        }
    ]
    
    with pytest.raises(ValueError, match="No readings left after applying 9AM-6PM Monday-Friday filter"):
        agent.process_readings(readings)

def test_process_readings_safe(agent):
    timestamps = pd.date_range(start="2023-10-23 09:00:00", end="2023-10-23 18:00:00", freq="1h")
    readings = []
    for ts in timestamps:
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"Tnw": 10.0, "Tg": 10.0}
        })
        
    result = agent.process_readings(readings)
    assert result == 0.0

def test_process_readings_risk(agent):
    timestamps = pd.date_range(start="2023-10-23 09:00:00", end="2023-10-23 18:00:00", freq="1h")
    readings = []
    for ts in timestamps:
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"Tnw": 22.5, "Tg": 22.5}
        })
        
    result = agent.process_readings(readings)
    assert result == pytest.approx(0.5, 0.01)

def test_process_readings_high_risk(agent):
    timestamps = pd.date_range(start="2023-10-23 09:00:00", end="2023-10-23 18:00:00", freq="1h")
    readings = []
    for ts in timestamps:
        readings.append({
            "timestamp": ts.isoformat(),
            "value": {"Tnw": 30.0, "Tg": 30.0}
        })
        
    result = agent.process_readings(readings)
    assert result == pytest.approx(2.0, 0.01)

def test_process_readings_calculation_accuracy(agent):
    # Test with varying values to check 95th percentile logic
    # 20 readings. 19 low, 1 high.
    # 95th percentile should be close to the high one.
    
    # Low risk: Tnw=15, Tg=15 -> WBGT=15 -> Risk=0 (clipped)
    # High risk: Tnw=25, Tg=35 -> WBGT = 0.7*25 + 0.3*35 = 17.5 + 10.5 = 28
    # Risk = (28 - 20) / 5 = 8/5 = 1.6
    
    timestamps = pd.date_range(start="2023-10-23 09:00:00", periods=20, freq="30min")
    readings = []
    
    # 19 low readings
    for i in range(19):
        readings.append({
            "timestamp": timestamps[i].isoformat(),
            "value": {"Tnw": 15.0, "Tg": 15.0}
        })
        
    # 1 high reading (at index 19)
    readings.append({
        "timestamp": timestamps[19].isoformat(),
        "value": {"Tnw": 25.0, "Tg": 35.0}
    })
    
    # With 20 values, 95th percentile is between the 19th and 20th sorted value.
    # Sorted values: [0, 0, ..., 0, 1.6] (19 zeros, 1 value of 1.6)
    # Quantile 0.95 method in pandas:
    # index = (N-1)*q = 19 * 0.95 = 18.05
    # It interpolates between index 18 (0.0) and index 19 (1.6).
    # Value = val[18] + (val[19] - val[18]) * 0.05
    # Value = 0 + (1.6 - 0) * 0.05 = 0.08
    
    result = agent.process_readings(readings)
    
    assert result == pytest.approx(0.08, 0.01)
