import numpy as np

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
        
        average_reading = np.mean(readings)
        return average_reading