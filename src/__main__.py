from agents.ExampleAgent import ExampleAgent
from agents.EUOSHANoiseAgent import EUOSHANoiseAgent
from agents.EUOSHATemperatureAgent import EUOSHATemperatureAgent
import json
import cherrypy

if __name__ == "__main__":
    agent = None
    config = {}

    with open("config.json", 'r') as file:
        config = json.load(file)

    agent_type = config.get("agent-type")

    if agent_type == "EU-OSHA-noise":
        agent = EUOSHANoiseAgent()
    elif agent_type == "EU-OSHA-temperature":
        agent = EUOSHATemperatureAgent()
    else:
        raise ValueError("Agent Type not recognized")

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})

    agent.run()