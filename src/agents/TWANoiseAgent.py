import pandas as pd
import numpy as np
from functools import reduce

from .BaseAgent import BaseSensorAgent

class TWANoiseAgent(BaseSensorAgent):    
    """
    Processes readings from noise sensor to compute the daily noise exposure level (NEL)
        - NEL is calculated as the 8-hour time-weighted average (TWA) of the noise levels during work hours (9AM-6PM, Monday-Friday).
        - The daily TWA is calculated by converting the decibel readings to energy, averaging the energy, and then converting back to decibels.
        - The daily WSI (workplace stress index) is a ratio of the calculated TWA to the limit

    The limit for TWA is set at 85dB, which is the threshold for noise-induced hearing loss according to EU (OSH) guidelines.
    """

    NEL_DB_THRESHOLD = 85

    def __init__(self, agent_type: str, agent_address: str, catalog_address: str) -> None:
        super().__init__(agent_type, agent_address, catalog_address)

        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["required"] = ["db"]
        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["properties"] = {
            "db": {"type": "number"}
        }

    def process_readings(self, readings: list) -> float:
        if not readings:
            raise ValueError("No readings provided.")

        readings_df = pd.DataFrame(readings)
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
        readings_df['date'] = readings_df['timestamp'].dt.date

        # Expand the 'value' dictionary into separate columns for db
        readings_df = pd.concat([readings_df.drop(['value'], axis=1), readings_df['value'].apply(pd.Series)], axis=1)

        # Filter for work shift readings (9AM-6PM Monday-Friday)
        start_time = pd.to_datetime("09:00").time()
        end_time = pd.to_datetime("18:00").time()
        mask = (readings_df['timestamp'].dt.time >= start_time) & \
               (readings_df['timestamp'].dt.time <= end_time) & \
               (readings_df['timestamp'].dt.dayofweek < 5)
        shift_readings_df = readings_df[mask]
        if shift_readings_df.empty:
            raise ValueError("No readings during work hours")

        # Convert dB to energy
        shift_readings_df['energy'] = 10 ** (shift_readings_df['db'] / 10)

        # Calculate daily average energy
        daily_exp_lvl = shift_readings_df \
                            .groupby('date') \
                            .aggregate({'energy': 'mean'})
        
        # Convert back to dB
        daily_exp_lvl['NEL'] = 10*np.log10(daily_exp_lvl['energy'])
        
        # Calculate daily WSI and global score
        daily_exp_lvl["WSI"] = daily_exp_lvl['NEL'] / self.NEL_DB_THRESHOLD
        global_score = daily_exp_lvl['WSI'].mean()

        return global_score