"""
User Agent implementation.
Note:
In user's messages:
- role = "user" records the action agent's response
- role = "assistant" records the user agent's response
"""
import os
import time
from copy import deepcopy
from openai import OpenAI
from typing import List, Dict, Any

SYSTEM_PROMPT_TRAVEL_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Do not speculate information not provided in the instructions. For example, if the agent asks for an order ID but it is not mentioned in the instructions, do not fabricate an order ID; instead, directly state that you do not remember or do not have it.
- When information confirmation is needed, decide whether to confirm based on the content in the Instruction.
- Do not repeat instruction content in the conversation; instead, express the same information in your own words.
- Keep the dialogue natural and maintain the user's personality as described in the instructions.
- If the goal in the instructions has been achieved, generate a separate line with the message 'finish conversation' to end the dialogue.
- If the Instruction requires booking a round-trip flight, you need to state the intention "Book a round-trip flight" at the very beginning.
"""

SYSTEM_PROMPT_BASE_EN = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Generate only one line of content each time to simulate the user's message.
- Do not reveal all instruction content at once. Only provide information needed for the current step.
- Ensure that all information needed for the current step is provided completely. For example, when adding a reminder, you need to provide the reminder's description, title, and time, etc.
- Do not speculate information not provided in the instructions. For example, if the Instruction does not directly specify takeout content, do not fabricate takeout content.
- When asked if you need further assistance, make sure whether all main tasks in the Instruction have been completed. If not, continue to provide the next step task to the agent.
- Names appearing in the Instruction are assumed to be the user's full names.
- When the agent asks which message to delete, follow the Instruction's requirements to delete the message.
- You cannot proactively offer help to the agent. Respond to the agent's questions as per the Instruction's requirements, and do not fabricate any information you do not know.
- If all tasks are completed, generate a separate line with the message 'finish conversation' to end the dialogue.
"""


def llm_inference(
    model: str, 
    provider: str,
    messages: List[Dict[str, Any]], 
    temperature: float = None,
    ) -> str:
    """Non-streaming LLM inference."""
    if provider == "openai":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    else:
        # add other provider support here
        raise ValueError(f"Invalid provider: {provider}.")
    if temperature is None:
        temperature = 0.001
    # add other provider support here
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=10000,
                n=1,
            )
            content = response.choices[0].message.content
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content
                reasoning_content = reasoning_content.strip()
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            return content

        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            
            retries += 1
            
    print(f"Failed to get response after {max_retries} retries.")
    return ''


class UserAgent:
    """User agent that simulates human user interactions with the action agent."""
    
    def __init__(self, model, provider):
        self.messages = None
        self.conversations = None
        self.model = model
        self.system_prompt = None
        self.provider = provider


    def get_init_reply(self, task, involved_classes):
        """Get initial user reply based on task and involved classes."""
        # Clear state
        self.messages = None
        self.conversations = None
        self.system_prompt = None
        # Set system prompt based on involved classes
        if "BaseApi" in involved_classes:
            system_prompt = SYSTEM_PROMPT_BASE_EN
        elif "Travel" in involved_classes:
            system_prompt = SYSTEM_PROMPT_TRAVEL_EN
        self.system_prompt = system_prompt
        self.conversations = []
        self.messages = [
            {"role": "system", "content": self.system_prompt.format(instruction=task)},
            {"role": "user", "content": "Is there anything you need help with today?"},
        ]
        # Get initial user content
        raw_response, user_content = self._infer()
        self.messages.append({"role": "assistant", "content": raw_response})
        user_content = f"{user_content}"
        self.conversations.append({"user": user_content})
        return user_content
        

        
    def user_step(self, agent_response):
        """Process agent response and return user reply."""
        self.messages.append({"role": "user", "content": agent_response})
        self.conversations.append({"agent": agent_response})
        raw_response, user_content = self._infer()
        user_content = f"{user_content}"
        self.messages.append({"role": "assistant", "content": raw_response})
        self.conversations.append({"user": user_content})
        return user_content
       
    
    def _infer(self):
        """Infer user response from LLM and extract content."""
        raw_response = llm_inference(
            model=self.model,
            messages=self.messages,
            provider=self.provider
        )
        if "</think>" in raw_response:
            user_content = raw_response.split("</think>")[1].strip()
        else:
            user_content = raw_response.strip()
        return raw_response, user_content
    
        
    def get_messages(self):
        """Return a deep copy of messages."""
        return deepcopy(self.messages)