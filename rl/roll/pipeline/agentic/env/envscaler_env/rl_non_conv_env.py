"""
Non-conversational RL environment.
"""
from .base_env import EnvScalerBaseEnv

class EnvScalerNonConvRLEnv(EnvScalerBaseEnv):
    """Non-conversational RL environment where termination is handled by action agent."""

    def __init__(self, mode, env_items_path=None, task_items_path=None):
        self.env_name = "envscaler_non_conversation_rl"
        super().__init__(mode=mode, env_items_path=env_items_path, task_items_path=task_items_path)
        
    def get_initial_observation(self, task_item: dict):
        """Return task description as initial observation."""
        return f"{task_item['task']}"

    def is_action_terminated(self, action: dict):
        """Terminate when action agent explicitly signals task completion."""
        if action["name"] == "chat_with_user":
            if 'Task Completed' not in action["arguments"]['content']:
                print('warning: Task Completed not in action["arguments"]["content"]')
            return True
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """Non-conversation mode does not require observation termination."""
        return False
