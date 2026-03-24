from agents.TWANoiseAgent import TWANoiseAgent
from agents.WBGTAgent import WBGTAgent
from agents.NVIRVentlationAgent import NVIRVentilationAgent
import json
import cherrypy
import os

if __name__ == "__main__":
    agent = None

    agent_type = os.getenv("AGENT_TYPE")
    if not agent_type:
        raise ValueError("AGENT_TYPE environment variable not set")
    
    catalog_address = os.getenv("CATALOG_ADDRESS")
    if not catalog_address:
        raise ValueError("CATALOG_ADDRESS environment variable not set")
    
    agent_address = os.getenv("AGENT_ADDRESS")
    if not agent_address:
        raise ValueError("AGENT_ADDRESS environment variable not set")
        

    if agent_type == "TWA-noise":
        agent = TWANoiseAgent(agent_type, agent_address, catalog_address)
    elif agent_type == "WBGT":
        agent = WBGTAgent(agent_type, agent_address, catalog_address)
    elif agent_type == "NVIR-ventilation":
        agent = NVIRVentilationAgent(agent_type, agent_address, catalog_address)

    else:
        raise ValueError("Agent Type not recognized")

    agent.run()