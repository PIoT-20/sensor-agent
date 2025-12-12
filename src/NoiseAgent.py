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
    Requires 'shift_start_time' and 'shift_end_time' in configuration.
    Inherits from BaseSensorAgent.
    """
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        if 'shift_start_time' not in self.config:
            raise ValueError("Configuration must include 'shift_start_time' parameter.")
        if 'shift_end_time' not in self.config:
            raise ValueError("Configuration must include 'shift_end_time' parameter.")

    def process_readings(self, readings: list) -> float:
        """
        Processes a list of sensor readings to compute the simple average.
        
        Input: List of sensor readings.
        Output: Average reading as a float.
        """
        if not readings:
            raise ValueError("No readings provided.")
        
        start_time = pd.to_datetime(self.config['shift_start_time']).time()
        end_time = pd.to_datetime(self.config['shift_end_time']).time()
        shift_T = (end_time.hour - start_time.hour) % 24

        readings_df = pd.DataFrame(readings)
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])

        if start_time <= end_time:
            mask = (readings_df['timestamp'].dt.time >= start_time) & (readings_df['timestamp'].dt.time <= end_time)
        else: # overnight shift
            mask = (readings_df['timestamp'].dt.time >= start_time) | (readings_df['timestamp'].dt.time <= end_time)

        shift_readings_df = readings_df[mask]

        shift_readings_df = shift_readings_df.copy()
        shift_readings_df['energy'] = 10 ** (shift_readings_df['value'] / 10)

        daily_exp_lvl_df = shift_readings_df \
            .groupby(shift_readings_df['timestamp'].dt.date) \
            .aggregate({"energy": "mean"})
        
        daily_exp_lvl_df['leq'] = 10*np.log10(daily_exp_lvl_df["energy"]*(shift_T/8)) # adjust to 8-hour TWA and turn back to dB

        interval_exp_lvl = 10*np.log10((10**(daily_exp_lvl_df['leq']/10)).mean())

        return float(interval_exp_lvl)