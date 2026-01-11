"""
Conversational RL environment.
"""
from .utils.user_agent import UserAgent, user_system_prompt
from .base_env import EnvScalerBaseEnv

class EnvScalerConvRLEnv(EnvScalerBaseEnv):
    """Conversational RL environment that uses UserAgent for multi-turn dialogue."""

    def __init__(self, mode, user_model, provider, env_items_path=None, task_items_path=None):
        self.user_agent = UserAgent(
            system_prompt=user_system_prompt,
            model=user_model,
            provider=provider
        )
        self.env_name = "envscaler_conversation_rl"
        super().__init__(mode=mode, env_items_path=env_items_path, task_items_path=task_items_path)

    def get_initial_observation(self, task_item: dict):
        """Get initial observation from user agent's first reply."""
        return self.user_agent.get_init_reply(task=task_item['task'])

    def is_action_terminated(self, action: dict):
        """Conversation mode does not rely on action for termination."""
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """Terminate when user sends ###STOP### message."""
        return action.get("name") == "chat_with_user" and "###STOP###" in str(observation)