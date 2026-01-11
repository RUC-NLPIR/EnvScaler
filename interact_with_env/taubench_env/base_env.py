"""
Base environment class for TauBench.
"""
import random
from copy import deepcopy
from typing import Any, Dict, Type, Union

from .envs.tool import Tool
from .envs.user import load_user
from .utils.hash_util import get_data_hash
from .utils.parse_util import parse_response, parse_action

from .tau_bench_types import (
    Action,
    RewardResult,
    RewardOutputInfo,
    RewardActionInfo,
    RESPOND_ACTION_NAME,
)


class TauBenchBaseEnv:
    """
    TauBench base environment class that encapsulates common logic.
    Subclasses (retail, airline) only need to implement data, tool, task, and rule loading methods.
    """
    def __init__(self, env_domain: str, mode: str, user_model: str, user_strategy: str, user_provider: str):
        super().__init__()
        self.env_domain = env_domain
        self.mode = mode

        # Cache configuration
        self.user_strategy = user_strategy
        
        # Initialize resources (database, tool set, task set, provided by subclasses)
        self.data = self.load_database()
        self.tools = self.load_all_tools()
        self.tasks = self.load_all_tasks()

        # Tool mapping: function name → tool class
        self.tools_map: Dict[str, Type[Tool]] = {tool.get_info()["function"]["name"]: tool for tool in self.tools}
        # Tool description list (provided to Agent)
        self.tools_info = [tool.get_info() for tool in self.tools]
        # terminate_tools: calling these tools immediately ends the trajectory
        self.terminate_tools = self.get_terminate_tools()

        # Load Wiki and rules
        self.env_introduction = self.load_wiki()
        self.rules = self.load_rules()


        # User simulation object, loaded based on strategy and model
        self.user = load_user(
            user_strategy=user_strategy, 
            model=user_model, 
            provider=user_provider,
        )

        self.actions = None  # Record of executed actions
        self.task = None  # Current task
        self.task_index = None  # Current task id

    # -------------------- Methods that must be implemented by subclasses --------------------
    def load_database(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def load_all_tools(self):
        raise NotImplementedError

    def load_all_tasks(self):
        raise NotImplementedError

    def load_wiki(self) -> str:
        raise NotImplementedError

    def load_rules(self):
        raise NotImplementedError

    def get_terminate_tools(self):
        raise NotImplementedError

    # -------------------- Public methods --------------------


    def reset(self, seed=None, task_index=None, user_content_bank=None):
        """
        Reset environment.
        
        Args:
            seed: Random seed
            task_index: Specify a task id
            user_content_bank: By default user_content is LLM-generated. If user_strategy is "human" and user_content_bank is not empty, 
                             each round pulls a user_query from user_content_bank
        """
        # Reload data (actions modify database, need to reload database)
        self.data = self.load_database()
        self.actions = []  # Clear history actions
        
        # Select task (specified or random)
        if seed is not None:
            random.seed(seed)
        if task_index is None:
            task_index = random.randrange(0, len(self.tasks))
        self.task = self.tasks[task_index]
        self.task_index = task_index

        # Reset user and get initial user_query
        if self.user_strategy == "human":
            assert user_content_bank is not None
            initial_observation = self.user.reset(
                instruction=self.task.instruction, 
                user_content_bank=user_content_bank
            )
        else:
            initial_observation = self.user.reset(instruction=self.task.instruction)

        # Return initial observation (user query) and task info
        observation=initial_observation
        info={"env_introduction": self.env_introduction, "tools": self.tools_info ,"task": self.task}

        return observation, info

    def step(self, action: Union[str, dict, Action]):
        """
        Execute one step of environment interaction.
        
        Args:
            action: Model's string output (Prompt) or structured output (Function Calling) or Action object
        
        Returns:
            observation: May be tool execution result or user response
            reward: Reward
            terminated: Whether task is completed
            truncated: Whether truncated
            info: Additional information
        """
        raw_response = deepcopy(action)
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {"action": raw_response}

        # String response needs additional parsing to struct_response
        if isinstance(raw_response, str):
            parse_success, struct_response = parse_response(raw_response)
            # Need to check if parse and convert succeeded (e.g., message content and tool_calls may both be empty)
            if not parse_success:
                observation = {"type": "user", "content": "Error: Failed to parse response to struct response."}
                truncated = True
                return observation, reward, terminated, truncated, info
            parse_success, action = parse_action(struct_response=struct_response)
            if not parse_success:
                observation = {"type": "user", "content": "Error: Message to Action Failed."}
                truncated = True
                return observation, reward, terminated, truncated, info

        # Action is LLM's structured output (Function Calling)
        elif isinstance(raw_response, dict):
            parse_success, action = parse_action(struct_response=raw_response)
            if not parse_success:
                observation = {"type": "user", "content": "Error: Message to Action Failed."}
                truncated = True
                return observation, reward, terminated, truncated, info
        
        # Action is Action object (ground truth)
        else:
            action = raw_response
        
        info.update({"action": action.__dict__})
        
        # Record action
        self.actions.append(action)

        # Execute based on action type
        # 1. RESPOND_ACTION_NAME: directly speak to user, observation is user's content
        if action.name == RESPOND_ACTION_NAME:
            observation = self.user.step(action.kwargs["content"])
            # print(f"USER: {observation}")
            # If user_content contains ###STOP###, end trajectory
            terminated = "###STOP###" in observation
            observation = {"type": "user", "content": observation}

        # 2. Tool call: observation is tool execution result
        elif action.name in self.tools_map:
            try:
                # Execute tool call
                tool_response = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:
                tool_response = f"Error: {e}"  # Catch tool call errors
            # print(f"TOOL: {observation}")
            # If tool call name is in terminate_tools, end trajectory
            if action.name in self.terminate_tools:
                terminated = True
            observation = {"type": "tool", "content": tool_response}
        
        # 3. Unknown action
        else:
            print(f"Unknown action {action.name}")
            observation = {"type": "user", "content": f"Unknown action {action.name}"}

        # Trajectory ends → calculate reward
        if terminated:
            reward_res = self.calculate_reward()
            reward = reward_res.reward
            
        # If trajectory ends or is truncated, return user_messages in info
        if terminated or truncated:
            user_messages = self.user.get_user_messages()
            info.update({"user_messages":user_messages})
        
        return observation, reward, terminated, truncated, info


    def calculate_reward(self) -> RewardResult:
        """
        Calculate reward (binary 0 or 1).
        
        Two main steps:
        1. Data modification correctness check: Compare data hash after agent execution with ground truth.
        2. Expected output check: Verify if agent's RESPOND actions contain required key information.
        """

        # Calculate hash of current data state (after agent execution)
        data_hash = get_data_hash(self.data)
        reward = 1.0  # Default reward is 1, set to 0 if errors found

        # Filter out RESPOND_ACTION (dialogue actions), keep only tool operation actions
        actions = [
            action for action in self.task.actions if action.name != RESPOND_ACTION_NAME
        ]

        # Check if data modification is correct
        # Restore data to initial state
        self.data = self.load_database()

        # Execute actions defined by task (ground truth actions)
        for action in self.task.actions:
            if action.name not in self.terminate_tools:
                self.step(action=action)
    
        # Data is now in state after executing correct actions
        # Get corresponding hash value
        gt_data_hash = get_data_hash(self.data)

        # Compare results: if different, agent's data modification is incorrect
        info = RewardActionInfo(
            r_actions=data_hash == gt_data_hash, 
            gt_data_hash=gt_data_hash
        )
        # Inconsistent, reward is 0
        if not info.r_actions:
            reward = 0.0

        # If task has expected outputs (text), check if outputs contain them
        if len(self.task.outputs) > 0:
            r_outputs = 1.0  # Assume all outputs match
            outputs = {}  # Record whether each output is found
            for output in self.task.outputs:
                found = False
                for action in self.actions:  # Traverse agent's actual execution actions
                    if (
                        action.name == RESPOND_ACTION_NAME
                        and output.lower()
                        in action.kwargs["content"].lower().replace(",", "")
                    ):
                        found = True
                        break
                outputs[output] = found
                # If not found, reward is 0
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            
            # Save detailed information of output check
            info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)

        # Package RewardResult and return
        return RewardResult(reward=reward, info=info, actions=actions)