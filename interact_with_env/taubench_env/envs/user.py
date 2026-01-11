import abc
from copy import deepcopy
from typing import Optional, List, Dict, Any, Union
from taubench_env.utils.user_llm_inference import llm_inference


# Abstract base class: User simulation environment
class BaseUserSimulationEnv(abc.ABC):
    metadata = {}

    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        """Receive the previous message from the AI agent, then return the next message from the user"""
        raise NotImplementedError


# 1. Human user simulation
class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: str, user_content_bank: List[str]) -> str:
        self.user_content_bank = user_content_bank
        return self.user_content_bank.pop(0)

    def step(self, content: str) -> str:
        return self.user_content_bank.pop(0)


# 2. LLM user simulation🌟
# In this messages, user records the response from the assistant, and assistant records the response from the LLM simulated user
class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.reset()

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Call LLM to generate the next user message"""
        res = llm_inference(
            model=self.model, 
            provider=self.provider,
            messages=messages,
        )
        if "</think>" in res:
            res = res.split("</think>")[1].strip()
        
        self.messages.append({"role": "assistant", "content": res})
        return res

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        """Build the system prompt, telling LLM how to play the user"""
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def reset(self, instruction: Optional[str] = None) -> str:
        # Reset the conversation, the default initial conversation of assistant is 'Hi! How can I help you today?'
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def step(self, content: str) -> str:
        """Add the response from the assistant, and generate the response from the user"""
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)
    
    def get_user_messages(self) -> List[Dict[str, Any]]:
        return deepcopy(self.messages)
    

# 3. LLM User + ReAct🌟
# In this messages, user records the response from the assistant, and assistant records the response from the LLM simulated user
class ReactUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(self, model: str, provider: str) -> None:
        super().__init__()
        self.messages: List[Dict[str, Any]] = []
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.reset()

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""

    def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
        """Call LLM to generate the next user message"""
        res = llm_inference(
            model=self.model, 
            provider=self.provider,
            messages=messages,
        )
        
        if "</think>" in res:
            res = res.split("</think>")[1].strip()
        
        self.messages.append({"role": "assistant", "content": res})
        return self.parse_response(res)

    def reset(self, instruction: Optional[str] = None) -> str:
        self.messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(instruction=instruction),
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        return self.generate_next_message(self.messages)

    def parse_response(self, response: str) -> str:
        if "###STOP###" in response:
            return "###STOP###"
        if "Thought:" in response:
            _, response = response.split("Thought:")
        if "User Response:" in response:
            _, response = response.split("User Response:")
            return response.strip()
        else:
            Warning(f"Invalid response format: {response}")
            return response.strip()

    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        return self.generate_next_message(self.messages)

    def get_user_messages(self) -> List[Dict[str, Any]]:
        return deepcopy(self.messages)


# Factory function: load the user environment with specified strategy
def load_user(
    user_strategy: str,
    model: Optional[str] = "gpt-4o",
    provider: Optional[str] = None,
) -> BaseUserSimulationEnv:
    """Return the corresponding class instance based on the strategy"""
    assert user_strategy in ["human", "llm", "llm_react"]
    assert provider in [None, "openai"]

    # user_strategy == "human"
    if user_strategy == "human":
        return HumanUserSimulationEnv()

    # user_strategy == "llm"
    elif user_strategy =="llm":
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return LLMUserSimulationEnv(model=model, provider=provider)
    
    # user_strategy == "llm_react"
    elif user_strategy == "llm_react":
        if model is None:
            raise ValueError("LLM user strategy requires a model")
        if provider is None:
            raise ValueError("LLM user strategy requires a model provider")
        return ReactUserSimulationEnv(model=model, provider=provider)

    raise ValueError(f"Unknown user strategy {user_strategy}")