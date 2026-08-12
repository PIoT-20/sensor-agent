import json
import os

import cherrypy
import pandas as pd
import requests as req
from agents.AgentInterface import AgentInterface
from jsonschema import ValidationError, validate


class AgentConnector:
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
                    },
                },
            }
        },
    }

    exposed = True

    def __init__(
        self,
        agent_type: str,
        agent_address: str,
        catalog_address: str,
        agent: AgentInterface,
    ) -> None:
        self.agent_type = agent_type
        self.agent_address = agent_address
        self.catalog = catalog_address
        self.agent = agent

        if not self.agent_type:
            raise ValueError("Agent type must be specified")
        if not self.agent_address:
            raise ValueError("Agent address must be specified")
        if not self.catalog:
            raise ValueError("Catalog address must be specified")

        self.SENSOR_DATA_SCHEMA["properties"]["readings"]["items"]["properties"][
            "value"
        ] = agent.get_readings_schema()

        cherrypy_conf = {
            "/": {
                "request.dispatch": cherrypy.dispatch.MethodDispatcher(),
                "tools.staticdir.root": os.path.abspath(os.getcwd()),
                "tools.sessions.on": True,
                "request.show_tracebacks": False,
                "error_page.default": lambda status, message, traceback, version: (
                    message
                ),
            },
        }
        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": 8080,
                "engine.autoreload.on": False,
            }
        )
        cherrypy.tree.mount(self, "/process_readings", cherrypy_conf)

        self.load_cfg()

    def run(self) -> None:
        self.register()

        cherrypy.engine.start()
        cherrypy.engine.block()

    def load_cfg(self):
        try:
            self.cfg = json.load(open("cfg.json", "r"))
        except FileNotFoundError:
            self.cfg = {}

    def save_cfg(self):
        json.dump(self.cfg, open("cfg.json", "w"))

    def register(self) -> None:
        data = {
            "service_type": f"agent_{self.agent_type}",
            "address": self.agent_address,
            "description": f"Agent for processing sensor data of type {self.agent_type}",
        }

        response = req.post(f"{self.catalog}/service", json=data)

        if response.status_code == 201 or response.status_code == 200:
            self.cfg["service_id"] = response.json().get("_id").get("$oid")
            self.save_cfg()
            print(f"Registered with id: {self.cfg['service_id']}")

        else:
            raise Exception(f"Failed to register service: {response.text}")

    @cherrypy.expose
    @cherrypy.tools.json_out()  # type: ignore
    @cherrypy.tools.json_in()  # type: ignore
    def POST(self):
        try:
            validate(instance=cherrypy.request.json, schema=self.SENSOR_DATA_SCHEMA)
            readings = cherrypy.request.json.get("readings", [])

            ret = self.calculate_quality_index(readings)

            return {"quality_index": ret}

        except ValidationError as ve:
            print(f"Validation Error: {ve.message}")
            raise cherrypy.HTTPError(400, f"Invalid input: {ve.message}")
        except ValueError as ve:
            print(f"Value Error: {ve}")
            raise cherrypy.HTTPError(400, f"{ve}")
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            raise cherrypy.HTTPError(500, str(e))

    def calculate_quality_index(self, readings) -> float:
        readings_df = pd.DataFrame(readings)
        readings_df["timestamp"] = pd.to_datetime(readings_df["timestamp"], utc=True)
        readings_df["date"] = readings_df["timestamp"].dt.date

        # Expand the 'value' dictionary into separate columns for each sensor reading
        readings_df = pd.concat(
            [
                readings_df.drop(["value"], axis=1),
                readings_df["value"].apply(pd.Series),
            ],
            axis=1,
        )

        return self.agent.process_readings(readings_df)
