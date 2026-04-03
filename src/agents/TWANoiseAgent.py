import pandas as pd
import numpy as np
from functools import reduce

from .AgentInterface import AgentInterface

class TWANoiseAgent(AgentInterface):    
    """
    Processes readings from noise sensor to compute the daily noise exposure level (NEL)
        - NEL is calculated as the 8-hour time-weighted average (TWA) of the noise levels during work hours (9AM-6PM, Monday-Friday).
        - The daily TWA is calculated by converting the decibel readings to energy, averaging the energy, and then converting back to decibels.
        - The daily WSI (workplace stress index) is a ratio of the calculated TWA to the limit

    The limit for TWA is set at 85dB, which is the threshold for noise-induced hearing loss according to EU (OSH) guidelines.
    """

    READINGS_VALUE_SCHEMA = {
        "type": "object",
        "required": ["db"],
        "properties": {
            "db": {"type": "number"}
        }
    }

    NEL_DB_THRESHOLD = 85

    def process_readings(self, readings_df: pd.DataFrame) -> float:
        if readings_df.empty:
            raise ValueError("No readings provided.")

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