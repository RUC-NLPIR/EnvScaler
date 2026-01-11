"""
Non-conversational SFT environment with task judgment.
Compared to RL environment:
1. For cost efficiency, SFT tasks do not synthesize verification functions,
   trajectories are collected only through LLM self-judgment ("Task Failed") and format filtering.
"""
import os
import json
import random
import traceback
from copy import deepcopy

from envscaler_env.utils.env_util import (
    init_env_class,
    init_env_instance,
    get_state_diff,
    get_state_info,
)
from envscaler_env.utils.parse_util import parse_response, parse_action




class EnvScalerNonConvSFTEnv:
    """
    Non-conversational SFT environment:
    - Manages task dataset and environment dataset loading
    - Provides reset/step workflow
    - Records trajectory (no reward calculation for SFT)
    """

    def __init__(self, mode, env_items_path=None, task_items_path=None):
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
        env_items_path = os.path.join(folder_path, "env_v1_85_brief.json")
        with open(env_items_path, encoding="utf-8") as f:
            env_items = json.load(f)

        print(f"Load {len(env_items)} envs from {env_items_path}!")
        return env_items


    def load_task_items(self):
        """Load task dataset."""
        folder_path = os.path.join(os.path.dirname(__file__), "data")

        if self.mode == "eval":
            task_items_path = os.path.join(folder_path, "all_pass_tasks_eval_148.json")
        elif self.mode == "train":
            task_items_path = os.path.join(folder_path, "task_v2_gpt5_2550_w_checklist.json")
        else:
            raise ValueError("mode must be eval or train")

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


    def reset(self, seed=None, task_index=None):
        """Reset environment and return initial observation + tool info + task info."""
        self.reset_attributes()

        if seed is not None:
            random.seed(seed)

        # Randomly select task or load specified task
        if task_index is None:
            task_index = random.randrange(0, len(self.task_items))

        self.task_item = self.task_items[task_index]
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

        info = deepcopy({"env_introduction": self.env_introduction, "tools": self.tools, "task": self.task_item})
        return init_observation, info


    # ==============================
    # Load environment and initialize instance
    # ==============================
    
    def load_env_and_instance(self, env_id: int, init_config: dict):
        """Load environment and initialize instance based on env_id."""
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
            reward = self.calculate_reward()
            self._record_step(action, observation, terminated, reward)
            
            if hasattr(self, "user_agent"):
                user_messages = self.user_agent.get_messages()
                info.update({"user_messages": user_messages})
            
            return observation, reward, terminated, truncated, info

        try:
            # Call environment method
            if action["name"] == "chat_with_user":
                observation = {"type": "user", "content": self.user_agent.user_step(agent_response=action['arguments']['content'])}
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
            
            if terminated or truncated:
                if hasattr(self, "user_agent"):
                    user_messages = self.user_agent.get_messages()
                    info.update({"user_messages": user_messages})
            
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
        if not (hasattr(self.env_instance, method_name) or method_name == "chat_with_user"):
            return False
        # Check action parameters
        params = action.get("arguments", {})
        # TODO: Check if action is one of the target tools
        if not isinstance(params, dict) or (method_name == "chat_with_user" and "content" not in params):
            return False
        return True
        

    def calculate_reward(self) -> float:
        """SFT data does not need reward, only rule-based filtering."""
        return 0.0

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
    # Termination and observation methods
    # ==============================

    def get_initial_observation(self, task_item: dict):
        """Return task description as initial observation."""
        return f"{task_item['task']}"

    def is_action_terminated(self, action: dict):
        """Terminate when action agent signals task completion or failure."""
        if action["name"] == "chat_with_user":
            if 'Task Completed' not in action["arguments"]['content'] and 'Task Failed' not in action["arguments"]['content']:
                print('warning: Task Completed and Task Failed are all not in action["arguments"]["content"]')
            return True
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """Non-conversation mode does not require observation termination."""
        return False