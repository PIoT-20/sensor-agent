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

    if agent_type == "TWA-noise":
        agent = TWANoiseAgent()
    elif agent_type == "WBGT":
        agent = WBGTAgent()
    elif agent_type == "NVIR-ventilation":
        agent = NVIRVentilationAgent()

    else:
        raise ValueError("Agent Type not recognized")

    agent.run()