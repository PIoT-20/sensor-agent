import pytest
import cherrypy
from unittest.mock import MagicMock
from agents.BaseAgent import BaseSensorAgent

class MockAgent(BaseSensorAgent):
    def process_readings(self, readings: list) -> float:
        # Simple mock implementation
        if not readings:
            return 0.0
        return sum(r['value']['v'] for r in readings) / len(readings)

@pytest.fixture
def agent():
    agent = MockAgent()
    # We need to mock cherrypy.request.json since POST accesses it directly
    return agent

def test_base_agent_creation(agent):
    assert agent is not None
    assert isinstance(agent, BaseSensorAgent)

def test_post_valid_data(agent):
    cherrypy.request = MagicMock()
    cherrypy.request.json = {
        "readings": [
            {"timestamp": "2023-10-23T10:00:00", "value": {"v": 10.0}},
            {"timestamp": "2023-10-23T11:00:00", "value": {"v": 20.0}}
        ]
    }
    
    result = agent.POST()
    assert result == {'quality_index': 15.0}

def test_post_invalid_schema(agent):
    cherrypy.request = MagicMock()
    # Missing "readings"
    cherrypy.request.json = {"data": []}
    
    with pytest.raises(cherrypy.HTTPError) as excinfo:
        agent.POST()
    
    assert excinfo.value.status == 400
    assert "Invalid input" in excinfo.value._message

def test_post_process_error(agent):
    cherrypy.request = MagicMock()
    cherrypy.request.json = {
        "readings": [
            {"timestamp": "2023-10-23T10:00:00", "value": {"v": 10.0}}
        ]
    }
    
    # Force process_readings to raise exception
    agent.process_readings = MagicMock(side_effect=Exception("Processing failed"))
    
    with pytest.raises(cherrypy.HTTPError) as excinfo:
        agent.POST()
        
    assert excinfo.value.status == 500
    assert "Processing failed" in excinfo.value._message
