"""
Automatically convert environment source code to gym/gem-like interactive environment.
"""
import types
import traceback
from copy import deepcopy


def get_state_diff(old_state: dict, new_state: dict) -> dict:
    """Compare two state dictionaries and return difference details."""
    diff_result = {}

    # Find union of all keys
    all_keys = set(old_state.keys()) | set(new_state.keys())

    for key in all_keys:
        old_val = old_state.get(key)
        new_val = new_state.get(key)

        if key not in old_state:
            # Added key
            diff_result[key] = {"added": new_val}
        elif key not in new_state:
            # Removed key
            diff_result[key] = {"removed": old_val}
        else:
            # Both exist → compare values
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recursively compare dictionaries
                sub_diff = get_state_diff(old_val, new_val)
                if sub_diff:  # Only record if there are changes
                    diff_result[key] = sub_diff
            else:
                # Simple type comparison
                if old_val != new_val:
                    diff_result[key] = {"changed": {"old":old_val, "new":new_val}}

    return diff_result



class InteractiveEnv:
    """Interactive environment wrapper with gym-like API."""

    def __init__(self, env_class, max_steps):
        self.env_class = env_class
        self.env_instance = None
        self.current_step = 0
        self.max_steps = max_steps
        self.trajectory = []  # Record trajectory for each step

    def env_init(self, init_config=None):
        """Initialize environment instance."""
        self.current_step = 0
        try:
            self.env_instance = self.env_class({})
        except Exception as e:
            self.env_instance = self.env_class()

        if init_config:
            self._apply_init_config(self.env_instance, init_config)
        # Reset trajectory (step 0 represents initial state)
        self.trajectory = [
                {
                    "step": 0, 
                    "state_snapshot": deepcopy(self.get_state_info())
                }
            ]

    def env_step(self, action: dict):
        """Execute one step of interaction. Returns: observation, reward, terminated, truncated, info."""
        self.current_step += 1
        terminated = False
        truncated = False
        info = {}

        # Exceeded maximum steps
        if self.current_step > self.max_steps:
            self._record_step(action, {}, 0, terminated, truncated, info)
            return {"error": "Max steps reached"}, 0, True, True, info

        try:
            method = getattr(self.env_instance, action["name"])
            # Check parameters
            if "params" not in action:
                return {"error": "No params in action"}, 0, terminated, truncated, info

            observation = method(**action.get("params", {}))

            # Termination condition: reached max steps
            if self.current_step == self.max_steps:
                truncated = True
                terminated = True

            # Calculate reward when task ends
            if terminated:
                reward = self.calculate_reward()
            else:
                reward = 0.0

            self._record_step(action, observation, reward, terminated, truncated, info)
            return observation, reward, terminated, truncated, info

        # Exception handling (internal environment exception)
        except Exception:
            error_log = traceback.format_exc()
            observation = {"error": "<Exception>\n" + error_log}
            # Exceptions usually indicate task end
            reward = 0.0
            terminated = True
            truncated = True
            info = {}
            self._record_step(action, observation, reward, terminated, truncated, info)
            return observation, reward, terminated, truncated, info

    def _record_step(self, action, observation, reward, terminated, truncated, info):
        """Record current step information to trajectory."""
        last_state = self.trajectory[-1]["state_snapshot"]
        current_state = deepcopy(self.get_state_info())
        state_diff = get_state_diff(last_state, current_state)
        step_record = {
            "step": self.current_step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "state_snapshot": current_state,
            "state_diff": state_diff,
        }
        self.trajectory.append(step_record)

    def _apply_init_config(self, env, init_config):
        """Apply initial configuration to environment instance."""
        for key, value in init_config.items():
            setattr(env, key, value)

    def get_state_info(self):
        """Return instance state (excluding __xxx__ built-in attributes)."""
        obj = self.env_instance
        state = {
            k: v
            for k, v in vars(obj).items()
            if not (k.startswith("__") and k.endswith("__"))
        }
        return state

    def calculate_reward(self):
        """Placeholder reward function: currently returns 0."""
        return 0


def build_env_from_str(env_str: str, class_name: str, max_steps: int) -> InteractiveEnv:
    """Build interactive environment from source code string."""
    module = types.ModuleType("dynamic_env")
    exec(env_str, module.__dict__)
    env_class = getattr(module, class_name)
    return InteractiveEnv(env_class, max_steps)