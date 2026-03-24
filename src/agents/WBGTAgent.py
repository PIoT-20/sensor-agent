import pandas as pd
import numpy as np

from .BaseAgent import BaseSensorAgent

class WBGTAgent(BaseSensorAgent):
    """
    Processes readings from 2 different temperature sensors to calculate the Wet Bulb Globe Temperature (WBGT) index, 
    which is a composite temperature used to estimate the effect of temperature, humidity, wind speed and solar 
    radiation on humans. The agent calculates a score based on the WBGT index:
        - WBGT: Wet Bulb Globe Temperature, calculated as 0.7*Tnw + 0.3*Tg, where:
            - Tnw: Natural Wet Bulb: thermometer covered by a wet cotton wick
            - Tg: Black Globe: A thermometer inside a matte black copper sphere

    The limit for WBGT is set at 25°C, which is the threshold for heat stress according to EU (OSH) guidelines.

    The daily WSI (workplace stress index) is calculated as a weighted average between the average WBGT and the 
    95th percentile WBGT during work hours (9AM-6PM, Monday-Friday), to capture both typical and worst-case conditions.

    The final score is the average of the daily WSI scores across all days in the window.
    """

    T_LIMIT = 25.0 # WBGT limit in °C according to EU (OSH) guidelines
    ALPHA = 0.4  # Weighting factor for average vs 95th percentile in final score aggregation
    
    def __init__(self, agent_type: str, agent_address: str, catalog_address: str) -> None:
        super().__init__(agent_type, agent_address, catalog_address)

        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["required"] = ["dB"]
        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["properties"] = {
            "Tnw": {"type": "number"},
            "Tg": {"type": "number"}
        }

    def process_readings(self, readings: list) -> float:
        if not readings:
            raise ValueError("No readings provided.")

        readings_df = pd.DataFrame(readings)
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
        readings_df['date'] = readings_df['timestamp'].dt.date

        # Expand the 'value' dictionary into separate columns for Tnw and Tg
        readings_df = pd.concat([readings_df.drop(['value'], axis=1), readings_df['value'].apply(pd.Series)], axis=1)

        # Filter for work shift readings (9AM-6PM Monday-Friday)
        start_time = pd.to_datetime("09:00").time()
        end_time = pd.to_datetime("18:00").time()
        mask = (readings_df['timestamp'].dt.time >= start_time) & \
               (readings_df['timestamp'].dt.time <= end_time) & \
               (readings_df['timestamp'].dt.dayofweek < 5)
        shift_readings_df = readings_df[mask]
        if shift_readings_df.empty:
            raise ValueError("No readings during work shift")

        # Calculate WBGT for each reading (WBGT = 0.7*Tnw + 0.3*Tg)
        shift_readings_df['WBGT'] = 0.7 * shift_readings_df['Tnw'] + 0.3 * shift_readings_df['Tg']

        # Weighted Aggregation (average and 95th percentile)
        def calc_wsi(x):
            avg = x.mean()
            p95 = x.quantile(0.95)
            wsi = ((self.ALPHA * avg) + ((1 - self.ALPHA) * p95)) / self.T_LIMIT
            return wsi
        
        # Group WBGT readings by date, calculate daily WSI, and then average across days for final score
        wsi_global_score = shift_readings_df \
                            .groupby('date') \
                            ["WBGT"] \
                            .apply(calc_wsi) \
                            .mean()

        return wsi_global_score