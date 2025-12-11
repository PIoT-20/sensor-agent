from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import cherrypy
import os

class BaseSensorAgent(ABC):
    """
    The core agent that handles connections, loops, and error handling.
    Developers must inherit from this class.
    """
    exposed = True

    def __init__(self, config_file: str) -> None:
        """Initializes connection parameters from config."""
        with open(config_file, 'r') as file:
            self.config = json.load(file)

        cherrypy_conf={
            '/':{
                'request.dispatch':cherrypy.dispatch.MethodDispatcher(),
                'tools.staticdir.root': os.path.abspath(os.getcwd()),
                'tools.sessions.on': True,
            },
            '/api': {
                'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            }
        }
        cherrypy.tree.mount(self, '/process_readings', cherrypy_conf)

    def run(self) -> None:
        cherrypy.engine.start()
        cherrypy.engine.block()
    
    def register(self) -> None:
        """Registers the agent within the IoT platform."""
        pass

    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def POST(self, *uri, **params):
        try:
            readings = cherrypy.request.json.get('readings', [])
            ret = self.process_readings(readings)
            return {'quality_index': ret}
        except Exception as e:
            raise cherrypy.HTTPError(500, str(e))

    @abstractmethod
    def process_readings(self, readings: list) -> float:
        """
        THE CONTRACT: The developer MUST implement this method.
        Input: Clean sensor readings.
        Output: A single quality metric (or result object).
        """
        pass