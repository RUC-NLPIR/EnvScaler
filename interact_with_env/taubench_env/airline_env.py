"""
Airline environment for TauBench.
"""
import os
import json
from .base_env import TauBenchBaseEnv


class TauBenchAirlineEnv(TauBenchBaseEnv):
    """Airline environment implementation for TauBench."""
    def __init__(self, mode, user_model, user_strategy, user_provider):
        super().__init__(env_domain="airline", mode=mode, user_model=user_model, user_strategy=user_strategy, user_provider=user_provider)

    def load_database(self):
        folder_path = os.path.join(os.path.dirname(__file__), "envs", "airline", "data")
        with open(os.path.join(folder_path, "flights.json"), encoding="utf-8") as f:
            flight_data = json.load(f)
        with open(os.path.join(folder_path, "reservations.json"), encoding="utf-8") as f:
            reservation_data = json.load(f)
        with open(os.path.join(folder_path, "users.json"), encoding="utf-8") as f:
            user_data = json.load(f)
        return {"flights": flight_data, "reservations": reservation_data, "users": user_data}

    def load_all_tools(self):
        from .envs.airline.tools import ALL_TOOLS
        return ALL_TOOLS

    def load_all_tasks(self):
        if self.mode == "eval":
            from .envs.airline.tasks_test import TASKS
            return TASKS
        else:
            raise "airline only support eval mode"

    def load_wiki(self):
        wiki_path = os.path.join(os.path.dirname(__file__), "envs", "airline", "wiki.md")
        with open(wiki_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_rules(self):
        from .envs.airline.rules import RULES
        return RULES

    def get_terminate_tools(self):
        return ["transfer_to_human_agents"]
