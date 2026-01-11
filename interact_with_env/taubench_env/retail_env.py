"""
Retail environment for TauBench.
"""
import os
import json
from .base_env import TauBenchBaseEnv


class TauBenchRetailEnv(TauBenchBaseEnv):
    """Retail environment implementation for TauBench."""
    def __init__(self, mode, user_model, user_strategy, user_provider):
        super().__init__(env_domain="retail", mode=mode, user_model=user_model, user_strategy=user_strategy, user_provider=user_provider)

    def load_database(self):
        folder_path = os.path.join(os.path.dirname(__file__), "envs", "retail", "data")
        with open(os.path.join(folder_path, "orders.json"), encoding="utf-8") as f:
            order_data = json.load(f)
        with open(os.path.join(folder_path, "products.json"), encoding="utf-8") as f:
            product_data = json.load(f)
        with open(os.path.join(folder_path, "users.json"), encoding="utf-8") as f:
            user_data = json.load(f)
        return {"orders": order_data, "products": product_data, "users": user_data}

    def load_all_tools(self):
        from .envs.retail.tools import ALL_TOOLS
        return ALL_TOOLS

    def load_all_tasks(self):
        if self.mode == "eval":
            from .envs.retail.tasks_test import TASKS_TEST
            return TASKS_TEST
        elif self.mode == "dev":
            from .envs.retail.tasks_dev import TASKS_DEV
            return TASKS_DEV
        elif self.mode == "train":
            from .envs.retail.tasks_train import TASKS_TRAIN
            return TASKS_TRAIN
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def load_wiki(self):
        wiki_path = os.path.join(os.path.dirname(__file__), "envs", "retail", "wiki.md")
        with open(wiki_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_rules(self):
        from .envs.retail.rules import RULES
        return RULES

    def get_terminate_tools(self):
        return ["transfer_to_human_agents"]
