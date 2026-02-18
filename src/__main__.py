from agents.EUOSHANoiseAgent import EUOSHANoiseAgent
from agents.EUOSHATemperatureAgent import EUOSHATemperatureAgent
import json
import cherrypy
import os

if __name__ == "__main__":
    agent = None

    agent_type = os.getenv("AGENT_TYPE")
    if not agent_type:
        raise ValueError("AGENT_TYPE environment variable not set")

    if agent_type == "EU-OSHA-noise":
        agent = EUOSHANoiseAgent()
    elif agent_type == "EU-OSHA-temperature":
        agent = EUOSHATemperatureAgent()
    else:
        raise ValueError("Agent Type not recognized")

    agent.run()