"""
BFCL Multi-Turn BaseEnvironment (only support base sub-category)
"""
import gem
import copy
import random
import json
import inspect
import importlib
from pathlib import Path
from gem import Env
from copy import deepcopy
from typing import List, Dict, Any, Callable, Union, Sequence, Tuple


from .file_util import load_file
from .xml_parser import XMLParser
from .tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH
from .bfcl_reward import BfclRewrard



class BfclEnv(gem.Env):
    def __init__(self, mode = "multi_turn_base"):
        super().__init__()
        # API class name to module path mapping
        self.CLASS_FILE_PATH_MAPPING = {
            "GorillaFileSystem": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.gorilla_file_system",
            "MathAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.math_api",
            "MessageAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.message_api",
            "TwitterAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.posting_api",
            "TicketAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.ticket_api",
            "TradingBot": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.trading_bot",
            "TravelAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.travel_booking",
            "VehicleControlAPI": "roll.pipeline.agentic.env.bfcl_env.bfcl_envs.vehicle_control",
        }

        # Stateless API classes (e.g., MathAPI doesn't need initialization)
        self.STATELESS_CLASSES = ["MathAPI"]

        # Track all environment instances (for parallel sessions with isolated API instances)
        self.env_instances = {}

        # Load dataset
        self.dataset, self.dataset_idx = self.load_dataset(mode=mode)

        # Initialize parsers and rewarder
        self.llm_parser = XMLParser(fields=["think", "tool_call"])
        self.rewarder = BfclRewrard()
        self.state = None
        self.env_name = "bfcl"


    def reset(self, seed=None, task_index=None):
        """Reset environment and sample a data instance."""
        Env.reset(self, seed)
        self.env_instances.clear()  # Clear environment instances
        self.env_instances = {}
        self.state = None

        if task_index is not None:
            # Fixed index selection
            sample = self.dataset[task_index]
        else:
            # Random sampling
            try:
                if seed is not None:
                    random.seed(seed)
                sample_idx = random.choice(self.dataset_idx)
                sample = self.dataset[sample_idx]
            except (RuntimeError, RuntimeWarning):
                next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
                return self.reset(next_seed)

        self.state = self._initialize_environments(sample)
        user_question = sample["user_first_question"]
        observation={"type": "user", "content": user_question}

        info = deepcopy({
            "env_name": "bfcl",
            "env_introduction": sample["env_introduction"], 
            "tools": sample["tools_info"], 
            "task": sample})
        return observation, info

    
    def step(self, action: str):
        """Execute one step of interaction with the environment."""
        info = {}
        truncated = False  # Truncation managed by env_manager, not here
        llm_response = action
        info.update({"action": llm_response})

        # Check if current trajectory is completed
        entry_completed = self.is_entry_completed(llm_response)

        if entry_completed:
            # Episode completed
            terminated = True
            observation = {"type": "user", "content": "Trajectory finished."}

            # Fill remaining ground truth tool calls
            while len(self.state["ground_truth_answer_bank"]) > 0:
                ground_truth_answer = self.state["ground_truth_answer_bank"].pop(0)
                _ = self.call_tool(tool_json=ground_truth_answer, ground_truth=True)

                if len(self.state["ground_truth_answer_bank"]) > 0:
                    self.state["successful_func_calls"].append([])

            # Check completeness
            assert len(self.state["successful_func_calls"]) == len(self.state["sample"]["answers"])

            reward, reward_info = self.get_reward()  # Calculate reward
            info.update({"reward_info": copy.deepcopy(reward_info)})

        else:
            # Episode not finished
            reward=0.0  # No intermediate rewards
            terminated=False

            # Check if current turn/sub-task is completed
            turn_completed=self.is_turn_completed(llm_response)

            if turn_completed:
                # If current turn/sub-task is completed, feedback is new sub-task/user_question
                next_user_question = self.state['user_question_bank'].pop(0)
                observation={"type": "user", "content": next_user_question}

                # Ground-truth corresponding tool calls also move forward one turn
                ground_truth_answer = self.state["ground_truth_answer_bank"].pop(0)
                _ = self.call_tool(tool_json=ground_truth_answer,ground_truth=True)

                # Append a new list to successful_func_calls for next turn
                self.state["successful_func_calls"].append([])

            else:
                # If current turn/sub-task is not completed, feedback is tool execution result
                try:
                    # 1. Parse assistant's last response (last message), try to parse "tool" field (XML or json extraction)
                    parsed = self.llm_parser.parse(llm_response)
                    # 2. If a usable "tool" field is actually obtained (meaning the function call instruction is captured in the model output)
                    if hasattr(parsed, 'tool_call') and parsed.tool_call is not None:
                        
                        info.update({"action": parsed.tool_call})
                        
                        # Call tool (API method), pass the parsed content to call_tool
                        result = self.call_tool(parsed.tool_call)

                        if len(result) > 0:
                            # If there is a non-empty result, return it to system message (XML wrapped result content)
                            tool_result = result
                        else:
                            # If execution failed, return error message
                            tool_result = "Error: Tool execution returned empty output."
                    else:
                        # If no function call is detected (tool field), give detailed error提示+操作建议
                        tool_result = "Error: Function call not found in current assistant response. Tool command must be one list of JSON objects. Please ensure correct formatting. If task is finished, please respond with 'Task Completed'."
                except Exception as e:
                    # Check if the exception is a normal unexpected case (e.g., environment code or gt data致命问题），continue to raise, no兜底
                    if "not expected" in str(e).lower():
                        raise Exception(f"Error in env_response is not expected!! Error: {e}")
                    # Parse XML format or json serialization (e.g., assistant output不合规范）情况下，提示错误
                    tool_result = f"Error: Invalid XML format: {str(e)}.  Tool command must be one list of JSON objects. Please ensure correct formatting."
                observation={"type": "tool", "content": tool_result}

        return observation, reward, terminated, truncated, info
    

    def call_tool(self, 
                  tool_json: str, 
                  ground_truth: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Call tool methods (API methods) in the environment and return execution results.
        
        Two modes:
        1. ground_truth=True: Execute standard answers from dataset (string-based API call sequence) using eval
        2. ground_truth=False: Execute model-generated tool calls (JSON list with {name, args} format)
        """
        # Ground truth tool call execution
        if ground_truth:
            try:
                if not isinstance(tool_json, list):
                    print(tool_json)
                    raise Exception("Error in ground truth tool execution is not expected!!")
                
                all_func_call_results = []
                # Build method name to class name mapping
                method_to_instance = {}
                for class_name, instance in self.state["ground_truth_environment"].items():
                    for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                        if not method_name.startswith('_'):
                            method_to_instance[method_name] = class_name

                # Process each ground truth call string
                for func_call in tool_json:
                    if "(" not in func_call:
                        print(tool_json)
                        print(func_call)
                        raise Exception(f"Error in ground truth tool execution is not expected!!")
                    # Extract method name
                    method_name = func_call.split("(")[0].strip()
                    if method_name not in method_to_instance:
                        print(tool_json)
                        print(func_call)
                        raise Exception(f"Error in ground truth tool execution is not expected!!")
                    
                    class_name = method_to_instance[method_name]
                    instance = self.state["ground_truth_environment"][class_name]
                    modified_call = f"self.state['ground_truth_environment']['{class_name}'].{func_call}"
                    
                    try:
                        result = eval(modified_call)
                        result_str = str(result) if result is not None else "Success"
                        all_func_call_results.append(f"Function Call {func_call} Succeeded. Result: {result_str}")
                    except Exception as e:
                        print(tool_json)
                        print(func_call)
                        raise Exception(f"Error in ground truth tool execution is not expected!!")
                # Return all call results as JSON
                return json.dumps(all_func_call_results)
            except Exception as e:
                print(tool_json)
                print(e)
                raise Exception(f"Error in ground truth tool execution is not expected!!")

        # Normal model output tool calls (JSON format)
        try:
            raw_tool_json = copy.deepcopy(tool_json)
            tool_json = tool_json.replace("arguments", "args")
            command = json.loads(tool_json)
            command = [command]
            all_func_call_results = []
            # Process tool calls sequentially, stop on first failure

            # Validate input is a list (multiple tool calls per turn possible)
            if not isinstance(command, list):
                all_func_call_results.append("Error: Invalid tool command. Tool command must be one list of JSON objects. Please ensure correct formatting.")
                return json.dumps(all_func_call_results)
            if command == []:
                all_func_call_results.append("Function Call Failed. Error: Found empty tool calls.")
                return json.dumps(all_func_call_results)

            # Process each tool call
            for tool_call in command:
                # Each call must have 'name' (method name) and 'args' (parameter dict)
                if not (isinstance(tool_call, dict) and "name" in tool_call and "args" in tool_call and isinstance(tool_call["args"], dict)):
                    all_func_call_results.append(f"Function Call {tool_call} Failed. Error: Tool command must be a dictionary with 'name' key and 'args' as a dictionary. Function calls after this will not be executed.")
                    return json.dumps(all_func_call_results)
                
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Convert list args to tuple to match API requirements
                for key, value in tool_args.items():
                    if isinstance(value, list):
                        tool_args[key] = tuple(value)
                tool_call["args"] = tool_args
                
                # Find method in environment instances
                found_method = False
                if self.env_instances == {}:
                    raise Exception(f"Environment is empty")
                for class_instance in self.state["environment"].values():
                    if hasattr(class_instance, tool_name):
                        found_method = True
                        tool_func = getattr(class_instance, tool_name)
                        break
                
                # Method not found: check available tools and report error
                if not found_method:
                    available_tools = []
                    for class_name in self.state['sample']['involved_classes']:
                        func_doc = load_file(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name])
                        for func in func_doc:
                            available_tools.append(func['name'])
                    if tool_name in available_tools:
                        # Tool in involved_classes but not instantiated: environment bug
                        print(f"Tool Name: {tool_name}")
                        print(f"Involved Classes: {self.state['sample']['involved_classes']}")
                        print(f"State Environment: {self.state['environment']}")
                        raise Exception(f"Error: Method '{tool_name}' found in involved classes but not found in any class instance. Available Tools: {available_tools}")
                    # Tool not in API list: return error message
                    all_func_call_results.append(f"Function Call {tool_call} Failed. Error: Method '{tool_name}' not found in any class instance. Function calls after this will not be executed.")
                    return json.dumps(all_func_call_results)
                
                # Execute the method
                try:
                    result = tool_func(**tool_args)
                except Exception as e:
                    all_func_call_results.append(f"Function Call {tool_call} Failed during execution. Error: {e}. Function calls after this will not be executed.")
                    return json.dumps(all_func_call_results)
                
                # Check if result contains error indication
                # NOTE: This below gives false negatives: sometimes the function return result has the word "error" in it
                if "'error':" in str(result).lower():
                    all_func_call_results.append(f"Function Call {tool_call} Failed during execution. Error: {result}. Function calls after this will not be executed.")
                    return json.dumps(all_func_call_results)
                
                # Success: record call and add to successful calls
                all_func_call_results.append(f"Function Call {tool_call} Succeeded. Result: {result}")
                self.state['successful_func_calls'][-1].append(tool_call)
            # Return results after all tools succeed
            if len(all_func_call_results) == 1:
                all_func_call_results = all_func_call_results[0]
                all_func_call_results = all_func_call_results.replace("args", "arguments")
            return json.dumps(all_func_call_results)

        # Handle invalid JSON input  
        except json.JSONDecodeError:
            import traceback
            traceback.print_exc()
            all_func_call_results = []
            all_func_call_results.append("Error in decoding tool call: Invalid JSON format. Tool command must be one list of JSON objects. Please ensure correct formatting.")
            return json.dumps(all_func_call_results)
        except Exception as e:
            print(tool_json)
            print(e)
            raise Exception(f"Error here is not expected!! Error: {e}")


    # Dataset loading
    def load_dataset(self, mode):
        """Load dataset for the specified mode."""
        assert mode in ["multi_turn_base"]
        BASE_DIR = Path(__file__).parent
        data_path = str(BASE_DIR / "data" / f"data_{mode}.json")
        raw_dataset=load_file(data_path)
        dataset_idx=[]
        dataset={}
        for raw_sample in raw_dataset:
            sample={}
            sample["id"]=raw_sample['id']
            sample["env_introduction"] = ""
            sample["tools_info"] = raw_sample["tools"]
            sample["questions"]=raw_sample["questions"]
            sample["user_first_question"]=sample["questions"][0]
            sample["answers"]=raw_sample["answers"]
            sample["initial_config"]=raw_sample["initial_config"]
            sample["involved_classes"]=raw_sample["involved_classes"]
            dataset[sample["id"]]=copy.deepcopy(sample)
            dataset_idx.append(sample["id"])
        
        print(f"Loading {mode} dataset from {data_path}, data_size {len(dataset_idx)}")
        return dataset, dataset_idx


    # Environment instance initialization
    def _initialize_environments(self,sample) -> None:
        """Initialize environment objects (API class instances) for each state instance."""
        state={}

        # Assign unique instance_id for each state (for parallel instances isolation)
        if "instance_id" not in state:
            state["instance_id"] = id(state)
        instance_id = state["instance_id"]

        state["sample"]=sample
        state["successful_func_calls"]=[[]]
        state["user_question_bank"]=copy.deepcopy(sample["questions"][1:])
        state["ground_truth_answer_bank"]=copy.deepcopy(sample["answers"])
        
        # Ensure instance container exists in global environment dict
        if instance_id not in self.env_instances:
            self.env_instances[instance_id] = {}
        
        # Determine involved API classes for this state (tools/simulated interfaces needed)
        involved_classes = sample["involved_classes"]

        # Initialize environment structure if not exists (separate for main/ground_truth/initial)
        if "environment" not in state:
            state["environment"] = {}
            state["ground_truth_environment"] = {}
            state["initial_environment"] = {}

        # Initialize each required API class
        for class_name in involved_classes:
            if class_name not in state["environment"]:
                # Get module path from class name
                module_name = self.CLASS_FILE_PATH_MAPPING[class_name]
                module = importlib.import_module(module_name)
                class_ = getattr(module, class_name)
                # Create three separate instances: main, ground truth (for scoring), initial snapshot
                class_instance = class_()
                ground_truth_class_instance = class_()
                initial_instance_copy = class_()
                # Load initial config for stateful APIs (classes with scenario state)
                if class_name not in self.STATELESS_CLASSES:
                    initial_config = sample["initial_config"]
                    class_initial_config = initial_config.get(class_name, {})
                    class_instance._load_scenario(copy.deepcopy(class_initial_config))
                    ground_truth_class_instance._load_scenario(copy.deepcopy(class_initial_config))
                    initial_instance_copy._load_scenario(copy.deepcopy(class_initial_config))

                # Register three instances in global and state environment dicts
                self.env_instances[instance_id][class_name] = {
                    'main': class_instance,
                    'ground_truth': ground_truth_class_instance,
                    'initial_instance': initial_instance_copy
                }
                state["environment"][class_name] = class_instance
                state["ground_truth_environment"][class_name] = ground_truth_class_instance
                state["initial_environment"][class_name] = initial_instance_copy
        return state
    
    # State checking
    def is_turn_completed(self, llm_response: str) -> bool:
        """Check if current turn/sub-task is completed."""
        if "<think>" in llm_response and "</think>" in llm_response:
            llm_response = llm_response.split("</think>")[1]
        
        return "TASK_FINISHED" in llm_response or "task_finished" in llm_response or "Finish" in llm_response or "Task Completed" in llm_response

    def is_entry_completed(self, llm_response: str):
        """Check if current episode is terminated (completed/exit/error)."""
        if "<think>" in llm_response and "</think>" in llm_response:
            llm_response = llm_response.split("</think>")[1]
        
        # Termination 1: All user questions exhausted and TASK_FINISHED tag present
        if len(self.state["user_question_bank"]) == 0 and self.is_turn_completed(llm_response):
            return True
        
        # Termination 2: TASK_ERROR/task_error tag found in assistant response
        if "<TASK_ERROR>" in llm_response or "task_error" in llm_response:
            return True
        
        return False

    # Reward calculation
    def get_reward(self):
        """Calculate trajectory reward."""
        try:
            reward, reward_info = self.rewarder.unified_reward_func(self.state)
        except Exception as e:
            print("error in get_reward: ", e)
            reward = 0.0
            reward_info = {
                "state_score": 0.0,
                "func_score": 0.0,
                "base_score": 0.0,
            }
        return reward, reward_info