import pandas as pd

from BaseAgent import BaseSensorAgent

class ExampleAgent(BaseSensorAgent):
    """
    An agent that processes sensor data to compute simple average of readings.
    Inherits from BaseSensorAgent.
    """

    def process_readings(self, readings: list) -> float:
        """
        Processes a list of sensor readings to compute the simple average.
        
        Input: List of sensor readings.
        Output: Average reading as a float.
        """
        if not readings:
            return 0.0
        
        readings_df = pd.DataFrame(readings)
        average_reading = readings_df['value'].mean()
        return average_reading