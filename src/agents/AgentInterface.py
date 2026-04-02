import pandas as pd

class AgentInterface():
    READINGS_VALUE_SCHEMA_EXAMPLE = {
        "type": "object",
        "required": ["reading_type1", "reading_type2"],
        "properties": {
            "reading_type1": {"type": "number"},
            "reading_type2": {"type": "number"}
        }
    }

    def get_readings_schema(self):
        if not hasattr(self, "READINGS_VALUE_SCHEMA"):
            raise NotImplementedError("You must define a READINGS_VALUE_SCHEMA for your agent.")
        else:
            return getattr(self, "READINGS_VALUE_SCHEMA")


    def process_readings(self, readings_df: pd.DataFrame) -> float:
        """
        THE CONTRACT: The developer MUST implement this method.
        """
        raise NotImplementedError("You must implement the process_readings method.")
    