"""
Multi-step environment.
"""
from .base_env import AceBenchBaseEnv

class AceBenchMultiStepEnv(AceBenchBaseEnv):
    """Multi-step environment where termination is handled by action agent."""
    
    def __init__(self, domain, truncated_steps):
        super().__init__(domain, truncated_steps)
        
    def get_initial_observation(self, task_item: dict):
        """Return task question as initial observation."""
        return f"{task_item['question']}"

    def is_action_terminated(self, action: str):
        """Terminate when action contains 'Task Completed'."""
        if "Task Completed" in action:
            return True
        return False

    def is_observation_terminated(self, observation: dict):
        """In multi-step mode, termination is handled by agent."""
        return False
    
    def get_chat_response(self, chat_content: str):
        """Multi-step mode has no agent-user interaction."""
        return "Please do not ask me any questions, use the known conditions to solve the problem"

    
    def get_env_introduction(self, involved_classes):
        """Get environment introduction (in original implementation, only provided in multi-turn mode)."""
        return ""