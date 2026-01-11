"""
Base environment class for AceBench.
"""
import json
import random
import importlib
import inspect
import copy
from copy import deepcopy
from typing import List, Dict
from acebench_env.calc_reward import calc_score
from acebench_env.utils.process_tool_schema import process_tool_schema


CLASS_FILE_PATH_MAPPING_EN = {
    "BaseApi": "acebench_env.acebench_envs.phone_platform.base_api",
    "MessageApi": "acebench_env.acebench_envs.phone_platform.message",
    "ReminderApi": "acebench_env.acebench_envs.phone_platform.reminder",
    "FoodPlatform": "acebench_env.acebench_envs.phone_platform.food_services",
    "Travel": "acebench_env.acebench_envs.travel",
}

SAVED_CLASS = {
                "BaseApi": ["wifi","logged_in"],
                "MessageApi": ["inbox"],
                "ReminderApi": ["reminder_list"],
                "FoodPlatform":["users","logged_in_users","orders"],
                "Finance":["user_accounts", "is_logged_in","deposit_history","withdrawal_history","loan_history","orders","holdings"],
                "Travel": ["users","reservations"],
               }

STATELESS_CLASSES = []

class AceBenchBaseEnv:
    """Base environment class for AceBench."""
    
    def __init__(self, domain: str, truncated_steps: int):
        self.truncated_steps = truncated_steps
        self.task_item = None
        # Load dataset
        self.dataset, self.dataset_idx = self.load_dataset(domain=domain)
        self.involved_instances: Dict[str, object] = {}
        self.class_method_name_mapping: Dict[str, List[str]] = {}
        # Record all environment instances
        self.env_instances = {}
        # Record history actions (only tool_calls, not user dialogue)
        self.history_tool_calls = []
        # Record current step count
        self.cur_step = 0

    def reset(self, seed=None, task_index=None):
        """Instantiate all involved environment classes and build method mapping."""
        self.involved_instances.clear()
        self.env_instances.clear()
        self.class_method_name_mapping.clear()
        self.history_tool_calls = []
        self.task_item = None
        self.cur_step = 0
        
        # Select test sample
        if task_index is not None:
            # Select sample by fixed index
            self.task_item = self.dataset[task_index]
        else:
            # Random sampling
                if seed is not None:
                    random.seed(seed)
                self.task_item = deepcopy(random.choice(self.dataset))
        
        # Task information
        initial_config = self.task_item["initial_config"]
        involved_classes = self.task_item["involved_classes"]
        test_id = self.task_item["id"].split("_")[-1]

        # Instantiate all involved environment classes and build method mapping
        for class_name in involved_classes:
            module_name = CLASS_FILE_PATH_MAPPING_EN[class_name]
            instance_name = f"{class_name.lower()}_instance"

            # Dynamically load class and create instance
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()

            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=False)
                base_config = initial_config.get("BaseApi", {})
                class_instance._load_scenario(copy.deepcopy(base_config), long_context=False)

            self.involved_instances[class_name] = instance_name

            # Build function name to instance mapping dictionary (for locating class instance by function name)
            for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
                if not method_name.startswith("_"):
                    self.class_method_name_mapping.setdefault(method_name, []).append(instance_name)

            # Register to class environment dictionary
            self.env_instances[instance_name] = class_instance
        
        # Get initial observation
        init_observation = self.get_initial_observation(task_item=self.task_item)
        
        info = {
            "task": {"id": self.task_item["id"], "question": self.task_item["question"], "involved_classes": self.task_item["involved_classes"]},
            "env_introduction": self.get_env_introduction(involved_classes),
            "tools": [{"type": "function", "function": process_tool_schema(func_info)} for func_info in self.task_item["function"]]
        }

        # Return initial observation (may be initial state snapshot)
        # return {"status": "env_reset", "classes": list(self.involved_instances.keys())}
        return init_observation, info
    
    def step(self, action: str | dict):
        action = deepcopy(action)
        self.cur_step += 1
        
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {"action": action}
        
        # Parse action into tool_calls and chat_content
        if isinstance(action, str):
            raise NotImplementedError("String action is not supported")

        elif isinstance(action, dict):
            tool_calls = [item["function"] for item in action["tool_calls"]]
            chat_content = action["content"]
        
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
            
        if not tool_calls and not chat_content:
            observation = {
                "type": "user",
                "content": "Action is empty."
            }
            return observation, reward, terminated, truncated, info
        
        # Record action to history
        if tool_calls:
            for tool_call in tool_calls:
                if not tool_call['arguments']:
                    tool_call['arguments'] = {}
                else:
                    if isinstance(tool_call['arguments'], str):
                        tool_call['arguments'] = json.loads(tool_call['arguments'])
                # Format as [func_name(parm1='xxx',parm2='xxx')] for 后续的评估
                def format_like_call_process(tool_call):
                    def format_value(v):
                        if isinstance(v, str):
                            return f"'{v}'"  # Wrap strings with single quotes
                        return str(v)  # Other types convert to str directly
                    return "[{}({})]".format(
                        tool_call['name'],
                        ', '.join(f"{k}={format_value(v)}" for k, v in tool_call['arguments'].items())
                    )
                tool_call_log = format_like_call_process(tool_call)
                # tool_call_log = '[' + tool_call['name'] + '(' + ', '.join(f"{k}={repr(v)}" for k, v in tool_call['arguments'].items()) + ')]'
                
                self.history_tool_calls.append(tool_call_log)
    
        # Check if action is termination action
        if self.is_action_terminated(chat_content):
            terminated = True
            observation = {"type": "user", "content": "Task finished"}
            
        else:
            # Get action response
            # If calling tools
            if tool_calls:
                execution_results = []
                # Execute tool calls
                for tool_call in tool_calls:
                    execution_result = self._get_func_call_response(func_name = tool_call["name"], func_args = tool_call["arguments"])
                    # print("execution_result:", execution_result)
                    execution_results.extend(execution_result)
                # Try to parse results as JSON
                parsed_results = []
                for item in execution_results:
                    try:
                        parsed_results.append(json.loads(item))
                    except json.JSONDecodeError:
                        parsed_results.append(item)
                        
                observation = {"type": "tool", "content": str(parsed_results)}
            # If chatting with user
            elif not tool_calls and chat_content:
                chat_response = self.get_chat_response(chat_content)
                observation = {"type": "user", "content": chat_response}
                
            # Check if observation is termination observation
            if self.is_observation_terminated(observation):
                terminated = True
        
        # Check if early stopping condition is met
        if self.is_truncated():
            truncated = True
        
        if terminated or truncated:
            reward, reward_info = self.calculate_reward()
            info.update({"reward_info": deepcopy(reward_info)})

        return observation, reward, terminated, truncated, info
    

    def load_dataset(self, domain: str):
        """Load test data files."""
        test_data_path = f"acebench_env/data/task/data_{domain}.json"
        answer_data_path = f"acebench_env/data/possible_answer/data_{domain}.json"
        task_data = []
        with open(test_data_path, 'r', encoding='utf-8') as file:
            task_data.extend(json.loads(line) for line in file)
        answer_data = []
        with open(answer_data_path, 'r', encoding='utf-8') as file:
            answer_data.extend(json.loads(line) for line in file)
        dataset_idx=[]
        dataset={}
        for task_case, answer_case in zip(task_data, answer_data):
            assert task_case["id"] == answer_case["id"]
            sample= {
                "id": task_case["id"],
                "question": task_case["question"],
                "function": task_case["function"],
                "initial_config": task_case["initial_config"],
                "involved_classes": task_case["involved_classes"],
                "ground_truth": answer_case["ground_truth"],
                "mile_stone": answer_case["mile_stone"]
            }
            dataset[sample["id"]]=copy.deepcopy(sample)
            dataset_idx.append(sample["id"])
        
        print(f"Loading {domain} dataset, data_size {len(dataset_idx)}")
        return dataset, dataset_idx
    
    def calculate_reward(self):
        """Calculate reward."""
        pred_env_states = [
            {
                env_name: self.get_env_instance_states(env_name, self.env_instances[env_instance_name])
            } 
                for env_name, env_instance_name 
                in self.involved_instances.items()
            ]
        pred_env_states = json.loads(json.dumps(pred_env_states))
        pred_actions = self.history_tool_calls
        target_env_states = self.task_item["ground_truth"]
        target_actions = self.task_item["mile_stone"]
        end_to_end_score, process_score, check_end_to_end_info, check_process_info =calc_score(pred_env_states,pred_actions,target_env_states,target_actions)
        return end_to_end_score, {"end_to_end_score": end_to_end_score, "process_score": process_score, "check_end_to_end_info": check_end_to_end_info, "check_process_info": check_process_info}
    

    def get_env_instance_states(self, env_name, env_instance):
        """Get final environment instance states."""
        state_dict = {}
        for item in env_instance.__dict__:
            if item in SAVED_CLASS[env_name]:
                state_dict[item] = env_instance.__dict__[item]
        return state_dict
    
    
    def _get_func_call_response(self, func_name: str, func_args: dict):
        """Get function call response based on function name and arguments."""
        execution_result = []
        # Note: Some class functions exist in both baseapi and subclasses due to inheritance
        if func_name in self.class_method_name_mapping:
            for instance_name in self.class_method_name_mapping[func_name]:
                func_args_str = '(' + ', '.join(f"{k}={repr(v)}" for k, v in func_args.items()) + ')'
                call_str = f"{instance_name}.{func_name}{func_args_str}"
                try:
                    func_call_result = eval(call_str, self.env_instances)
                    if isinstance(func_call_result, dict):
                        try:
                            tool_response = json.dumps(func_call_result)
                        except Exception:
                            tool_response = str(func_call_result)
                    else:
                        tool_response = str(func_call_result)
                except Exception as e:
                    tool_response = f"Error during execution: {str(e)}"
                if tool_response not in execution_result:
                    execution_result.append(tool_response)
        if len(execution_result) > 1:
            import pdb; pdb.set_trace()
        if not execution_result:
            print("Patch mechanism")
            tool_response = "No matching function found"
            execution_result.append(tool_response)
        return execution_result
    
    
    def is_truncated(self):
        """Only consider step-based truncation."""
        return self.cur_step >= self.truncated_steps

        
    
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
    
    def get_env_introduction(self, involved_classes):
        """Get environment introduction."""
        raise NotImplementedError
    

    def is_action_terminated(self, action: dict):
        """
        Termination request initiated by Action Agent.
        Common in single-turn multi-step scenarios where termination is handled by action agent.
        """
        raise NotImplementedError

    def is_observation_terminated(self, observation: dict):
        """
        Termination information returned by environment observation.
        Common in multi-turn multi-step scenarios where termination is handled by environment (user initiates).
        """
        raise NotImplementedError
    
    def get_chat_response(self, chat_content: str):
        """Get chat response."""
        raise NotImplementedError