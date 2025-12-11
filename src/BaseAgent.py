from abc import ABC, abstractmethod
from typing import Dict, Any
import json

class BaseSensorAgent(ABC):
    """
    The core agent that handles connections, loops, and error handling.
    Developers must inherit from this class.
    """

    def __init__(self, config_file: str) -> None:
        """Initializes connection parameters from config."""
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        pass

    def run(self) -> None:
        """
        The MAIN LOOP (Hidden from developer).
        1. Connects to source.
        2. Loops: _fetch_data() -> process_data() -> _publish_result()
        """
        pass
    
    def register(self) -> None:
        """Registers the agent within the IoT platform."""
        pass

    def rec_post(self, payload) -> Dict:
        """Receives POST requests with data payloads."""
        readings = payload.get('readings', [])
        ret = self.process_readings(readings)
        return {'result': ret}

    @abstractmethod
    def process_readings(self, readings: list) -> float:
        """
        THE CONTRACT: The developer MUST implement this method.
        Input: Clean sensor readings.
        Output: A single quality metric (or result object).
        """
        pass