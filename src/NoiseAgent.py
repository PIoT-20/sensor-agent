import pandas as pd
import numpy as np
from functools import reduce

from BaseAgent import BaseSensorAgent

def average_decibel(series):
    return reduce(lambda x, y: pow(10, 0.1*x) + pow(10, 0.1*y), series)

class EUOSHANoiseAgent(BaseSensorAgent):    
    """
    An agent that processes noise sensor data to compute the daily noise exposure level.
    It is the 8-hour time-weighted average (TWA).
    Inherits from BaseSensorAgent.
    """
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)

        self.SENSOR_DATA_SCHEMA["readings"]["items"]["properties"]["value"]["required"] = ["dB"]
        self.SENSOR_DATA_SCHEMA["readings"]["items"]["properties"]["value"]["properties"] = {
            "dB": {"type": "number"}
        }

    def process_readings(self, readings: list) -> float:
        """
        Processes a list of sensor readings to compute the simple average.
        
        Input: List of sensor readings.
        Output: Average reading as a float.
        """
        if not readings:
            raise ValueError("No readings provided.")

        readings_df = pd.DataFrame(readings)
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
        readings_df = pd.concat([readings_df.drop(['value'], axis=1), readings_df['value'].apply(pd.Series)], axis=1)

        start_time = pd.to_datetime("09:00").time()
        end_time = pd.to_datetime("18:00").time()
        mask = (readings_df['timestamp'].dt.time >= start_time) & \
               (readings_df['timestamp'].dt.time <= end_time) & \
               (readings_df['timestamp'].dt.dayofweek < 5)
        shift_readings_df = readings_df[mask]

        shift_readings_df = shift_readings_df.copy()
        shift_readings_df['energy'] = 10 ** (shift_readings_df['dB'] / 10)

        daily_exp_lvl_df = shift_readings_df \
            .groupby(shift_readings_df['timestamp'].dt.date) \
            .aggregate({"energy": "mean"})
        
        daily_exp_lvl_df['leq'] = 10*np.log10(daily_exp_lvl_df["energy"]) # turn back to dB

        interval_exp_lvl = 10*np.log10((10**(daily_exp_lvl_df['leq']/10)).mean())

        return float(interval_exp_lvl)