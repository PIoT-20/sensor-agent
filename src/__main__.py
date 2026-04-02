import json
import cherrypy
import os
from AgentConnector import AgentConnector
from agents.AgentFactory import AgentFactory

if __name__ == "__main__":
    agent_type = os.getenv("AGENT_TYPE")
    if not agent_type:
        raise ValueError("AGENT_TYPE environment variable not set")
    
    catalog_address = os.getenv("CATALOG_ADDRESS")
    if not catalog_address:
        raise ValueError("CATALOG_ADDRESS environment variable not set")
    
    agent_address = os.getenv("AGENT_ADDRESS")
    if not agent_address:
        raise ValueError("AGENT_ADDRESS environment variable not set")
        

    agent = AgentFactory.create_agent(agent_type)
    connector = AgentConnector(agent_type=agent_type, agent_address=agent_address, catalog_address=catalog_address, agent=agent)
    connector.run()