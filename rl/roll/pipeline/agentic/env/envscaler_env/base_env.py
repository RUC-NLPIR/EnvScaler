"""
Base environment class for EnvScaler.
"""
import os
import gem
import json
import random
import traceback
from gem import Env
from copy import deepcopy

from .utils.env_util import (
    init_env_class,
    init_env_instance,
    get_state_diff,
    get_state_info,
    run_check_function,
)
from .utils.parse_util import parse_response, parse_action


class EnvScalerBaseEnv(gem.Env):
    """
    Base environment class:
    - Manages task dataset and environment dataset loading
    - Provides reset/step workflow
    - Records trajectory and calculates rewards
    Subclasses must implement abstract methods (construct prompt, initial observation, termination conditions, etc.)
    """

    def __init__(self, mode, env_items_path=None, task_items_path=None):
        super().__init__()
        self.mode = mode

        # Load task dataset and environment dataset
        if task_items_path is not None:
            self.task_items = json.load(open(task_items_path, encoding="utf-8"))
            print(f"Ignore the mode {self.mode}.\nLoad task_items from {task_items_path}, total {len(self.task_items)} tasks!")
        else:
            self.task_items = self.load_task_items()
        if env_items_path is not None:
            self.env_items = json.load(open(env_items_path, encoding="utf-8"))
            print(f"Load env_items from {env_items_path}, total {len(self.env_items)} envs!")
        else:
            self.env_items = self.load_env_items()

        # Initialize logs and environment state
        self.reset_attributes()

    # ==============================
    # Data loading methods
    # ==============================

    def load_env_items(self):
        """Load environment dataset."""
        folder_path = os.path.join(os.path.dirname(__file__), "data")
        env_items_path = os.path.join(folder_path, "191_env_metadata.json")
        with open(env_items_path, encoding="utf-8") as f:
            env_items = json.load(f)

        print(f"Load {len(env_items)} envs from {env_items_path}!")
        return env_items


    def load_task_items(self):
        """Load task dataset."""
        folder_path = os.path.join(os.path.dirname(__file__), "data")

        if self.mode == "train":
            task_items_path = os.path.join(folder_path, "envscaler_rl_scenario_metadata.json")
        else:
            raise ValueError("mode must be train")

        with open(task_items_path, encoding="utf-8") as f:
            task_items = json.load(f)

        print(f"Load {len(task_items)} tasks from {task_items_path}!")
        return task_items

    # ==============================
    # State reset
    # ==============================

    def reset_attributes(self):
        """Reset class attributes (logs and environment state)."""
        # Log related
        self.current_step = 0
        self.trajectory = []

        # Environment related
        self.env_item = None
        self.env_class = None
        self.env_instance = None
        self.system_prompt = None

        # Scenario related (initial state, task item, check functions, etc.)
        self.init_config = None
        self.init_state = None
        self.pred_final_state = None
        self.task_item = None
        self.checklist_with_func = None


    def reset(self, seed=None, task_index=None):
        """Reset environment and return initial observation + tool info + task info."""
        Env.reset(self, seed)
        self.reset_attributes()

        if seed is not None:
            random.seed(seed)

        # Randomly select task or load specified task
        if task_index is None:
            if len(self.task_items) == 0:
                raise ValueError(f"No tasks available for mode '{self.mode}'")
            task_index = random.randrange(0, len(self.task_items))
        else:
            # Validate task_index
            if task_index < 0 or task_index >= len(self.task_items):
                raise ValueError(f"Invalid task_index {task_index}, expected range [0, {len(self.task_items)})")

        self.task_item = self.task_items[task_index]
        self.checklist_with_func = self.task_item["checklist_with_func"]
        self.task_id = self.task_item["task_id"]
        self.env_id = self.task_item["env_id"]
        self.init_config = self.task_item["init_config"]

        # Load environment and instance
        self.load_env_and_instance(env_id=self.env_id, init_config=self.init_config)
        # Construct system prompt
        self.env_introduction = self.construct_env_introduction(env_item=self.env_item)
        # Get tool list
        self.tools = deepcopy(self.env_item["tools"])
        # Get initial observation
        init_observation = self.get_initial_observation(task_item=self.task_item)
        init_observation = {"type": "user", "content": init_observation}

        info = deepcopy({"env_name": self.env_name, "env_introduction": self.env_introduction, "tools": self.tools, "task": self.task_item})
        return init_observation, info


    # ==============================
    # Load environment and initialize instance
    # ==============================
    
    def load_env_and_instance(self, env_id: int, init_config: dict):
        """Load environment and initialize instance based on env_id."""
        # Validate env_id
        if env_id < 0 or env_id >= len(self.env_items):
            raise ValueError(f"Invalid env_id {env_id}, expected range [0, {len(self.env_items)})")
        self.env_item = self.env_items[env_id]
        env_class_code = self.env_item["env_class_code"]
        env_class_name = self.task_item["env_class_name"]
        # Initialize environment class and instance
        self.env_class = init_env_class(env_class_code, env_class_name)
        self.env_instance = init_env_instance(self.env_class, init_config)
        # Save initial state
        self.init_state = get_state_info(self.env_instance)
        # Initial trajectory record
        self.trajectory.append({
            "step": 0,
            "state_snapshot": deepcopy(self.init_state)
        })

    # ==============================
    # Environment interaction step
    # ==============================

    def step(self, action: str | dict):
        """Execute one step of environment interaction, return observation, reward, terminated, truncated, info."""
        raw_response = deepcopy(action)
        
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {"action": raw_response}

        # Parse response to action dict
        # String response needs additional parsing to struct_response
        if isinstance(raw_response, str):
            parse_success, struct_response = self._parse_response(text_response=raw_response)
            if not parse_success:
                observation = {"type": "user", "content": "Error: Failed to parse response to struct response"}
                self._record_step(action, observation, terminated, reward)
                return observation, reward, terminated, truncated, info
        else:
            struct_response = raw_response
        
        parse_success, action = self._parse_action(struct_response)
        if not parse_success:
            observation = {"type": "user", "content": "Error: Failed to parse response to action"}
            self._record_step(action, observation, terminated, reward)
            return observation, reward, terminated, truncated, info
    
        info.update({"action": action})
        
        # Check action validity
        if not self.check_vaild_action(action=action):
            observation = {"type": "user", "content": "Error: Invalid action"}
            self._record_step(action, observation, terminated, reward)
            return observation, reward, terminated, truncated, info

        # Check if action is termination action
        if self.is_action_terminated(action):
            observation = {"type": "user", "content": "Task finished"}
            terminated = True
            self.pred_final_state = get_state_info(self.env_instance)
            reward = self.calculate_reward(self.checklist_with_func, self.init_state, self.pred_final_state)
            self._record_step(action, observation, terminated, reward)
            return observation, reward, terminated, truncated, info

        try:
            # Call environment method
            if action["name"] == "chat_with_user":
                # 检查user_agent是否已初始化（仅在ConvCustomEnv中初始化）
                if not hasattr(self, "user_agent") or self.user_agent is None:
                    raise AttributeError("user_agent is not initialized. chat_with_user action requires EnvScalerConvRLEnv.")
                observation = {"type": "user", "content": self.user_agent.user_step(agent_response=action["arguments"]["content"])}
            else:
                observation = {"type": "tool", "content": f"{getattr(self.env_instance, action['name'])(**action['arguments'])}"}
            
            # Check if observation is termination observation
            if self.is_observation_terminated(action, observation):
                terminated = True
                # Once finished, record final state snapshot and calculate reward
                self.pred_final_state = get_state_info(self.env_instance)
                reward = self.calculate_reward(self.checklist_with_func, self.init_state, self.pred_final_state)

            # Record and return
            self._record_step(action, observation, terminated, reward)
            return observation, reward, terminated, truncated, info

        except Exception:
            # Catch execution exception and terminate
            error_log = traceback.format_exc()
            observation = {"type": "user", "content": "Error: <Exception>\n" + error_log}
            terminated = True
            self._record_step(action, observation, terminated, reward)
            return observation, reward, terminated, truncated, info

    # ==============================
    # Utility methods
    # ==============================
    
    def _record_step(self, action, observation, terminated, reward):
        """Record current step trajectory."""
        last_state = self.trajectory[-1]["state_snapshot"]
        current_state = get_state_info(self.env_instance)
        state_diff = get_state_diff(last_state, current_state)
        # Increment step counter
        self.current_step += 1
        self.trajectory.append({
            "step": self.current_step,
            "action": action,
            "observation": observation,
            "terminated": terminated,
            "reward": reward,
            "state_snapshot": current_state,
            "state_diff": state_diff
        })
    

    def _parse_response(self, text_response: str):
        """Parse LLM output (string format) to action format."""
        # Try to parse structured action from raw_response
        parse_success, struct_response = parse_response(text_response)
        return parse_success, struct_response

    def _parse_action(self, struct_response: dict):
        """Parse struct_response to action."""
        parse_success, action = parse_action(struct_response)
        return parse_success, action
    
    
    def check_vaild_action(self, action: dict):
        """Check action validity."""
        # Check action name (must be environment method or chat_with_user)
        method_name = action.get("name")
        if not method_name:
            return False
        if not (hasattr(self.env_instance, method_name) or method_name == "chat_with_user"):
            return False
        # Check action parameters
        params = action.get("arguments", {})
        if not isinstance(params, dict) or (method_name == "chat_with_user" and "content" not in params):
            return False
        return True
        

    def calculate_reward(self, checklist_with_func: list, init_state: dict, pred_final_state: dict) -> float:
        """Calculate reward based on final state."""
        checklist_with_func_result = []

        for check_item in checklist_with_func:
            check_func_str = check_item["check_func"]
            success, result, error = run_check_function(
                func_code=check_func_str,
                init_state=init_state,
                final_state=pred_final_state
            )
            new_check_item = deepcopy(check_item)
            new_check_item["check_func_result"] = {"success": success, "result": result, "error": error}
            checklist_with_func_result.append(new_check_item)

        check_avg_result = round(
            sum([item["check_func_result"]["result"] for item in checklist_with_func_result if item["check_func_result"]["result"] is not None])
            / len(checklist_with_func_result),
            4
        )

        return check_avg_result

    def construct_env_introduction(self, env_item: dict):
        """Return environment introduction."""
        # Environment introduction
        env_brief_intro = env_item["environment_introduction"]
        # Environment rules
        env_rule_str = ""
        for rule in env_item.get("constraints_rules", []):
            env_rule_str += "- " + rule + "\n"
        env_introduction = f"# Environment Information\n\n## Brief Introduction:  \n{env_brief_intro}\n\n## Environment Rules / Constraints:  \n{env_rule_str}"
        return env_introduction


    # ==============================
    # Abstract methods (to be implemented by subclasses)
    # ==============================

    def get_initial_observation(self, task_item: dict):
        """
        Get initial observation. Generally:
        - Single-turn multi-step: initial observation is the task
        - Multi-turn multi-step: initial observation is user's initial dialogue
        """
        raise NotImplementedError

    def is_action_terminated(self, action: dict):
        """
        Termination request initiated by Action Agent.
        Common in single-turn multi-step scenarios where termination is handled by action agent.
        """
        raise NotImplementedError

    def is_observation_terminated(self, action: dict, observation: str):
        """
        Termination information returned by environment observation.
        Common in multi-turn multi-step scenarios where termination is handled by environment (user initiates).
        """
        raise NotImplementedError