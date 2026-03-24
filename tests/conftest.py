import sys
import os

# Add the src directory to sys.path so we can import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

os.environ["AGENT_TYPE"] = "test"
os.environ["AGENT_ADDRESS"] = "http://localhost:8080"
os.environ["CATALOG_ADDRESS"] = "http://localhost:9090"
