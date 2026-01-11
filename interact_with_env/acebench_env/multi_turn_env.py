"""
Multi-turn environment.
"""
from .base_env import AceBenchBaseEnv
from .utils.user_agent import UserAgent
from .acebench_envs.env_introduction import TRAVEL_PROMPT_EN, BASE_PROMPT_EN

class AceBenchMultiTurnEnv(AceBenchBaseEnv):
    """Multi-turn environment that uses UserAgent for agent-user dialogue."""
    
    # def __init__(self, domain):
    def __init__(self, domain, truncated_steps, user_model, user_provider):
        self.user_agent = UserAgent(
            model=user_model,
            provider=user_provider
        )
        super().__init__(domain, truncated_steps)
        
    def get_env_introduction(self, involved_classes):
        """Get environment introduction (only used in multi-turn mode)."""
        env_introduction = ""
        if "Travel" in involved_classes:
            env_introduction += TRAVEL_PROMPT_EN
        if "BaseApi" in involved_classes:
            env_introduction += BASE_PROMPT_EN
        return env_introduction

    def get_initial_observation(self, task_item: dict):
        """Get initial observation from user agent's first reply."""
        return self.user_agent.get_init_reply(task=task_item['question'], involved_classes=task_item["involved_classes"])
    
    def get_chat_response(self, chat_content: str):
        """Get chat response from user agent."""
        return self.user_agent.user_step(chat_content)

    def is_action_terminated(self, action: dict):
        """Conversation mode does not rely on action for termination."""
        return False

    def is_observation_terminated(self, observation: dict):
        """Terminate when user sends 'finish conversation' message."""
        return observation["type"] == "user" and "finish conversation" in observation["content"]