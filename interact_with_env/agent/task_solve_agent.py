"""
Task solving agent for interactive environments.
"""
from copy import deepcopy
from agent.system_prompt_util import conversational_system_prompt, non_conversational_system_prompt,  merge_tools_into_system_prompt
from agent.agent_llm_inference import llm_inference_fc, llm_inference_prompt


class TaskSolveAgent:
    """Agent that solves tasks in interactive environments using LLM inference."""
    def __init__(
        self,
        env_name,
        env,
        model,
        provider,
        temperature,
        infer_mode,
        max_steps,
        enable_thinking,
        api_key=None,
        base_url=None
    ):
        self.env_name = env_name
        self.env = env

        # LLM settings
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        assert infer_mode in ["prompt", "fc"]  # prompt: tool use via prompts, fc: tool use via function calling interface
        self.infer_mode = infer_mode
        self.enable_thinking = enable_thinking

        # Runtime settings
        self.max_steps = max_steps

        # Runtime state
        self.messages = []  # Conversation history
        self.current_observation = None
        self.current_info = None
        self.total_reward = 0.0
        self.terminated = False  # Environment terminated normally
        self.truncated = False  # Environment truncated
        self.step_count = 0

        # Trajectory recording
        self.trajectory = []  # Detailed execution information for each step

    def reset(self, task_index=None):
        """Reset environment and conversation history."""
        # Reset environment and get initial observation
        observation, info = self.env.reset(task_index=task_index)

        # Get environment introduction and available tools, build system prompt
        self.tools = info["tools"]
        self.user_tools = info.get("user_tools", [])
        # Select system prompt based on conversation/non-conversation mode
        if self.env_name in ["tau_bench_retail", "tau_bench_airline", "envscaler_conversation_rl", "envscaler_conversation_sft", "conv_custom_wo_reward", "acebench_multi_turn"] :
            system_prompt = conversational_system_prompt
        elif self.env_name in ["envscaler_non_conversation_rl", "envscaler_non_conversation_sft","bfcl", "acebench_multi_step"]:
            system_prompt = non_conversational_system_prompt 
        else:
            raise RuntimeError(f"Unknown env_name: {self.env_name}")  
        
        # Add environment introduction to system prompt if available
        if "env_introduction" in info and info["env_introduction"]:
            system_prompt = f"{system_prompt}\n\nThe following is an introduction to the current environment:\n{info['env_introduction']}"       
        
        # In prompt mode, merge tool information into system prompt
        if self.infer_mode == "prompt":
            system_prompt = merge_tools_into_system_prompt(system_prompt=system_prompt, tools=info["tools"])

        # Extract task info for logging
        task_item = deepcopy(info["task"])
        if self.env_name in ["tau_bench_retail", "tau_bench_airline"]:
            self.task_info = {"user_id": task_item.user_id, "instruction": task_item.instruction}
        elif self.env_name in ["envscaler_non_conversation_rl", "envscaler_conversation_rl", "envscaler_non_conversation_sft", "envscaler_conversation_sft"]:
            self.task_info = {"env_id": task_item["env_id"], "task_id": task_item["task_id"], "task": task_item["task"]}
        elif self.env_name in ["bfcl"]:
            self.task_info = {"id": task_item["id"], "questions": task_item["questions"], "involved_classes": task_item["involved_classes"]}
        elif self.env_name in ["acebench_multi_step", "acebench_multi_turn"]:
            self.task_info = {"id": task_item["id"], "question": task_item["question"], "involved_classes": task_item["involved_classes"]}
        else:
            raise RuntimeError(f"Unknown env_name: {self.env_name}")


        # Initialize message history
        self.messages = [{"role": "system", "content": system_prompt}]
        self.current_observation = deepcopy(observation)
        self.current_info = deepcopy(info)
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.user_messages = None
        self.step_count = 0
        self.trajectory = []

        # Add initial observation to messages (user role)
        self.messages.append({"role": "user", "content": observation})

        # Record initial trajectory information
        self.trajectory.append({
            "step": self.step_count,
            "observation": observation,
            # "info": info,
        })

        return observation, info

    def step(self):
        """
        Execute one environment step:
        1) Call LLM with current messages to generate response
        2) Pass LLM response as action to env.step
        3) Update conversation history, accumulated reward, trajectory, etc.
        Returns: (observation, reward, terminated, truncated, info, action)
        """

        # Check if environment has already finished
        if self.terminated or self.truncated:
            raise RuntimeError("Environment already finished. Please reset before calling step again.")

        # Call LLM inference (response parsing is done in env)
        if self.infer_mode == "prompt":
            # Prompt mode: returns LLM text (str)
            raw_response = llm_inference_prompt(
                provider=self.provider,
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking,
                api_key=self.api_key,
                base_url=self.base_url
            )
            if "</think>" in raw_response:
                raw_response = raw_response.split("</think>")[-1].strip()
        else:
            # FC mode: returns (reasoning_content, tool_calls, content) as dict
            raw_response = llm_inference_fc(
                provider=self.provider,
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                tools=self.tools,
                enable_thinking=self.enable_thinking,
                api_key=self.api_key,
                base_url=self.base_url
            )

        # Add model output to conversation history
        if self.infer_mode == "prompt":
            message = {"role": "assistant", "content": raw_response}
        else:
            message = {"role": "assistant", "content": raw_response["content"]}
            if raw_response["tool_calls"]:
                message["tool_calls"] = raw_response["tool_calls"]
            if raw_response["reasoning_content"]:
                message["reasoning_content"] = raw_response["reasoning_content"]
        self.messages.append(message)

        # Execute one environment step
        if raw_response == '':  # Special handling for empty response
            print("raw_response is empty, please check the model")
            observation, reward, terminated, truncated, info = "action is empty, please check the model", 0, True, True, {"action": ""}
        else:
            observation, reward, terminated, truncated, info = self.env.step(action=raw_response)
            action = info["action"]

        # Update internal state
        self.step_count += 1
        self.total_reward += float(reward or 0.0)
        self.current_observation = observation
        self.current_info = info
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        # Add new observation to conversation history
        if observation["type"] == "tool":
            # Prompt mode: tool role not allowed
            if self.infer_mode == "prompt":
                observation_content = f"<tool_response>\n{observation['content']}\n</tool_response>"  # Qwen3 tool response template style
                self.messages.append({"role": "user", "content": observation_content})
            # FC mode: tool role is allowed
            else: 
                if len(raw_response["tool_calls"]) < 1:
                    import pdb; pdb.set_trace()
                    print("!!!!!! raw_response['tool_calls'] is empty, please check the model")
                    self.messages.append({"role": "user", "content": observation['content']})
                else:
                    # TODO: support for multiple function calls
                    tool_call_id = raw_response["tool_calls"][0]['id']
                    tool_call_name = raw_response["tool_calls"][0]['function']['name']
                    self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_call_name, "content": observation['content']})
        else:
            self.messages.append({"role": "user", "content": observation["content"]})

        # Record trajectory
        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            # "info": info,
        })
        if "user_messages" in info:
            self.user_messages = info["user_messages"]
        return observation, reward, terminated, truncated, info, action

    def run(self, task_index=None, max_steps=None):
        """
        Run agent from reset, looping step until:
        - terminated is True, or
        - truncated is True, or
        - max_steps is reached
        Returns execution result dictionary.
        """
        max_steps = max_steps if max_steps is not None else self.max_steps

        # Initialize environment
        self.reset(task_index=task_index)

        # Loop execution
        while (not self.terminated) and (not self.truncated) and (self.step_count < max_steps):
            self.step()

        # Aggregate results
        result = {
            "task_info" : self.task_info,
            "tools": self.tools,
            "messages": self.messages,
            "user_messages": self.user_messages,
            "trajectory": self.trajectory,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_observation": self.current_observation,
            "final_info": self.current_info,
            "steps": self.step_count,
        }

        return result