from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import cherrypy
import os
from jsonschema import validate, ValidationError

class BaseSensorAgent(ABC):
    """
    The core agent that handles connections, loops, and error handling.
    Developers must inherit from this class.
    """

    POST_BODY_SCHEMA = {
        "type": "object",
        "required": ["readings"],
        "properties": {
            "readings": {
                "type": "array", 
                "items": {
                    "required": ["timestamp", "value"],
                    "type": "object", "properties": {
                        "timestamp": {"type": "string", "format": "date-time"},
                        "value": {"type": "number"},
                    }
                }
            }
        }
    }

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
            'request.show_tracebacks': False,
            'error_page.default': lambda status, message, traceback, version: message
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

    @cherrypy.tools.json_out() # type: ignore
    @cherrypy.tools.json_in() # type: ignore
    def POST(self, *uri, **params):
        try:
            validate(instance=cherrypy.request.json, schema=self.POST_BODY_SCHEMA)
            readings = cherrypy.request.json.get('readings', [])
            ret = self.process_readings(readings)
            return {'quality_index': ret}
        
        except ValidationError as ve:
            raise cherrypy.HTTPError(400, f"Invalid input: {ve.message}")
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