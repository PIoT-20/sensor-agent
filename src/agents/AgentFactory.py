from .AgentInterface import AgentInterface
from .NVIRVentlationAgent import NVIRVentilationAgent
from .TWANoiseAgent import TWANoiseAgent
from .WBGTAgent import WBGTAgent

class AgentFactory:
    """
    Factory class to create agent instances based on the agent type.
    """
    @staticmethod
    def create_agent(agent_type) -> AgentInterface:
        if agent_type == "NVIRVentilationAgent":
            return NVIRVentilationAgent()
        elif agent_type == "TWANoiseAgent":
            return TWANoiseAgent()
        elif agent_type == "WBGTAgent":
            return WBGTAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")