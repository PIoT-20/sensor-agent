import pandas as pd
import numpy as np

from .BaseAgent import BaseSensorAgent

class NVIRVentlationAgent(BaseSensorAgent):
    """
    Processes readings from 3 sensors (temperature, pressure, CO2) to calculate a ventilation score based on the
    Near-Field Ventilation Index Ratio (NVIR).

    The three sensors are:
        - Temperature (T): Air temperature in °C
        - Pressure (P): Air pressure in hPa
        - CO2 concentration (CO2): raw CO2 concentration in ppm

    The raw CO2 reading is adjusted using temperature and pressure to account for sensor calibration and environmental 
    factors, following the ideal gas law.
    The readings are then adjusted for sensor shifts (using daily baselines), and smoothed to reduce noise.

    The daily WSI (workplace stress index) is calculated as a weighted average between the average CO2 level and the 
    95th percentile during work hours (9AM-6PM, Monday-Friday), to capture both typical and worst-case conditions.
    The index is normalized by the PPM limit
    The final score is the average of the daily WSI scores across all days in the window.

    The limit is based on the REHVA guidelines, with 1000 ppm CO2 as the threshold for poor ventilation.
    """

    # Calibration constants (datasheet)
    P_CAL = 1013.25 
    T_CAL_K = 298.15

    PPM_LIMIT = 1000  # CO2 concentration above which ventilation is considered poor
    ALPHA = 0.4  # Weighting factor for average vs 95th percentile in final score aggregation
    
    
    def __init__(self) -> None:
        super().__init__()
        
        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["required"] = ["dB"]
        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"]["value"]["properties"] = {
            "T": {"type": "number"},
            "P": {"type": "number"},
            "CO2": {"type": "number"},
        }

    def adjust_readings(self, df):
        # 1. Sensor Calibration
        # Convert raw CO2 to ppm using the ideal gas law and datasheet calibration constants        
        df['Tk'] = df['T'] + 273.15
        df['CO2_adj'] = df['CO2'] * (self.P_CAL / df['P']) * (df['Tk'] / self.T_CAL_K)
        
        # 2. Daily Daily Baseline Correction (ABC)
        # The sensor drift and environmental factors can cause the baseline ("fresh air") CO2 to shift over time. 
        # To correct for this, we use a daily ABC:
        # 1) Get 5th percentile of CO2 each day to represent 'fresh air' 
        # 2) Shift the baseline so the daily 'clean air' is ~400ppm (normal outdoor level)
        daily_min = df \
                    .groupby('date') \
                    ['CO2_adj'] \
                    .transform(lambda x: x.quantile(0.05))
        
        df['CO2_adj'] = df['CO2_adj'] - (daily_min - 400)
        
        # 3. Smoothing (EWMA) - Low pass filter to reduce noise
        df['CO2_adj'] = df['CO2_adj'].ewm(alpha=0.1).mean()
        
        return df

    def process_readings(self, readings: list) -> float:
        if not readings:
            raise ValueError("No readings provided.")

        readings_df = pd.DataFrame(readings)
        readings_df['timestamp'] = pd.to_datetime(readings_df['timestamp'])
        readings_df['date'] = readings_df['timestamp'].dt.date

        # Expand the 'value' dictionary into separate columns for T, P, and CO2
        readings_df = pd.concat([readings_df.drop(['value'], axis=1), readings_df['value'].apply(pd.Series)], axis=1)

        # Apply adjustments to the raw readings (calibration, baseline correction, smoothing)
        adj_readings_df = self.adjust_readings(readings_df)

        # Filter for work shift readings (9AM-6PM Monday-Friday)
        start_time = pd.to_datetime("09:00").time()
        end_time = pd.to_datetime("18:00").time()
        mask = (adj_readings_df['timestamp'].dt.time >= start_time) & \
               (adj_readings_df['timestamp'].dt.time <= end_time) & \
               (adj_readings_df['timestamp'].dt.dayofweek < 5)
        shift_adj_readings_df = adj_readings_df[mask]


        # Weighted Aggregation (average and 95th percentile)
        def calc_wsi(x):
            avg = x.mean()
            p95 = x.quantile(0.95)
            wsi = ((self.ALPHA * avg) + ((1 - self.ALPHA) * p95)) / self.PPM_LIMIT
            return wsi
        
        # Group CO2 adjusted readings by date, calculate daily WSI, and then average across days for final score
        wsi_global_score = shift_adj_readings_df \
                            .groupby('date') \
                            ["CO2_adj"] \
                            .apply(calc_wsi) \
                            .mean()

        return wsi_global_score
        