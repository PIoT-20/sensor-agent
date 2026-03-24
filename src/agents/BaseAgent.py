from abc import ABC, abstractmethod
from typing import Dict, Any
import requests as req
import cherrypy
import os
from jsonschema import validate, ValidationError

class BaseSensorAgent(ABC):
    """
    The core agent that handles connections, loops, and error handling.
    Developers must inherit from this class.
    """

    SENSOR_DATA_SCHEMA = {
        "type": "object",
        "required": ["readings"],
        "properties": {
            "readings": {
                "type": "array", 
                "items": {
                    "type": "object", 
                    "required": ["timestamp", "value"],
                    "properties": {
                        "timestamp": {"type": "string", "format": "date-time"},
                        "value": {"type": "object"},
                    }
                }
            }
        }
    }

    exposed = True

    def __init__(self, agent_type: str, agent_address: str, catalog_address: str) -> None:
        self.agent_type = agent_type
        self.agent_address = agent_address
        self.catalog = catalog_address

        if not self.agent_type:
            raise ValueError("Agent type must be specified")
        if not self.agent_address:
            raise ValueError("Agent address must be specified")
        if not self.catalog:
            raise ValueError("Catalog address must be specified")

        cherrypy_conf={
            '/':{
            'request.dispatch':cherrypy.dispatch.MethodDispatcher(),
            'tools.staticdir.root': os.path.abspath(os.getcwd()),
            'tools.sessions.on': True,
            'request.show_tracebacks': False,
            'error_page.default': lambda status, message, traceback, version: message
            },
        }
        cherrypy.config.update({'server.socket_host': '0.0.0.0', "server.socket_port": 8080})
        cherrypy.tree.mount(self, '/process_readings', cherrypy_conf)

    def run(self) -> None:
        cherrypy.engine.start()
        cherrypy.engine.block()

        self.register()
    
    def register(self) -> None:
        data = {
            "service_type": "agent",
            "address": self.agent_address,
            "description": f"Agent for processing sensor data of type {self.agent_type}",
            "agent_type": self.agent_type
        }

        req.post(f"http://{self.catalog}/service", json=data)

    @cherrypy.expose
    @cherrypy.tools.json_out() # type: ignore
    @cherrypy.tools.json_in() # type: ignore
    def POST(self, *uri, **params):
        try:
            validate(instance=cherrypy.request.json, schema=self.SENSOR_DATA_SCHEMA)
            readings = cherrypy.request.json.get('readings', [])
            
            ret = self.process_readings(readings)
            return {'quality_index': ret}
        
        except ValidationError as ve:
            print(f"Validation Error: {ve.message}")
            raise cherrypy.HTTPError(400, f"Invalid input: {ve.message}")
        except ValueError as ve:
            print(f"Value Error: {ve}")
            raise cherrypy.HTTPError(400, f"{ve.message}")
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            raise cherrypy.HTTPError(500, str(e))

    @abstractmethod
    def process_readings(self, readings: list) -> float:
        """
        THE CONTRACT: The developer MUST implement this method.
        Input: Clean sensor readings.
        Output: A single quality metric (or result object).
        """
        pass