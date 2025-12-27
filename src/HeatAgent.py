import pandas as pd
import numpy as np

from BaseAgent import BaseSensorAgent

class EUOSHAHeatAgent(BaseSensorAgent):
    """
    An agent that processes readings from 2 different temperature sensors:
        - Tnw: Natural Wet Bulb: thermometer covered by a wet cotton wick
        - Tg: Black Globe: A thermometer inside a matte black copper sphere
    
    Inherits from BaseSensorAgent.
    """
    
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        
        self.SENSOR_DATA_SCHEMA["readings"]["items"]["properties"]["value"]["required"] = ["dB"]
        self.SENSOR_DATA_SCHEMA["readings"]["items"]["properties"]["value"]["properties"] = {
            "Tnw": {"type": "number"},
            "Tg": {"type": "number"}
        }

    def process_readings(self, readings: list) -> float:
        """
        Uses only the 9AM-6PM Monday-Friday readings.
        Calculates WBGT of each readings, scores readings according to 20°-25° range:
            - WBGT<20° => 0
            - 20°<WBGT<25° => propotional to deviation and between 0 and 1
            - WBGT>25° => propotional to deviation and higher than 1
        Calculates 95 percentile WBGT (worst cases)
        
        Input: List of sensor readings.
        Output: 95 percentile WBGT
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


        baseline = 20.0  # Score 0
        limit = 25.0     # Score 1

        shift_readings_df['WBGT'] = 0.7 * shift_readings_df['Tnw'] + 0.3 * shift_readings_df['Tg']

        shift_readings_df['risk_ratio'] = (shift_readings_df['WBGT'] - baseline) / (limit - baseline)
        shift_readings_df['risk_ratio'] = shift_readings_df['risk_ratio'].clip(lower=0)

        score = shift_readings_df['risk_ratio'].quantile(0.95)

        return float(score)