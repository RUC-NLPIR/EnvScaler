"""
User Agent implementation.
Note:
In user's messages:
- role = "user" records the action agent's response
- role = "assistant" records the user agent's response
"""
import re
from copy import deepcopy
from envscaler_env.utils.user_llm_inference import llm_inference

# --------------------------------------------------------------------
# System Prompt (User Agent)
# --------------------------------------------------------------------
user_system_prompt = \
"""You are a real human user interacting with an Agent assistant.  
Your current task is to have the Agent accomplish the following goal:  

[Task Goal]  
{task}  

**Core Principles:**  
- Do not directly or fully repeat the exact task instruction in your dialogue; instead, progress toward the goal gradually through multiple exchanges.  
- Deliver the task information in parts during the conversation so the Agent can slowly understand and move closer to the final objective.  
- When the task goal has been achieved, output a standalone message: `###STOP###` in your reply to end the dialogue. Do not include anything else.  

**Rules:**  
1. If the task contains multiple sub-tasks, do not reveal all of them at once; provide relevant sub-tasks one by one as the Agent asks.  
2. If completing the task requires multiple pieces of information, do not disclose them all at once; provide partial information in response to the Agent's questions.  
3. All requests must remain strictly within the scope of the task—do not add extra requirements, intentions, or invent information that was not part of the original task.  
4. Always keep the conversation focused on progressing toward the task, ensuring every sub-task or goal is covered and none are skipped.  

**Fidelity and Consistency Requirements:**  
- Always remain faithful to the original task wording throughout the conversation. Pay special attention to preserving exact **keywords, names, and proper nouns**—do not rephrase or alter them.  
- If the Agent assistant presents you with multiple options, only choose those that match the intent and constraints of the original task. If none fit, politely refuse and restate your requirement.  
- Do **not** introduce any new information that is not present in the original task description.  
- Do not repeat information you have already provided earlier in the conversation unless the Agent explicitly asks for clarification.  

**Style Requirements:**  
- Keep the dialogue natural and conversational, avoiding overly rigid or formal expressions.  

**Output Format (must be strictly followed):**  
# Thought:  
<Your thought process (this will NOT be sent to the Agent)> 
# Reply:  
<Your natural, conversational reply as the user, to be sent to the Agent>"""


class UserAgent:
    """User agent that simulates human user interactions with the action agent."""
    
    def __init__(self, system_prompt, model, provider, api_key=None, base_url=None):
        self.messages = None
        self.conversations = None
        self.model = model
        self.system_prompt = system_prompt
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url


    def get_init_reply(self, task):
        """Get initial user reply based on task."""
        self.conversations = []
        self.messages = [
            {"role": "system", "content": self.system_prompt.format(task=task)},
            {"role": "user", "content": "[Agent] Hi! How can I help you today?"},
        ]
        # Get initial user content
        raw_response, user_content = self._infer()
        self.messages.append({"role": "assistant", "content": raw_response})
        user_content = f"{user_content}"
        self.conversations.append({"user": user_content})
        return user_content
        

    def user_step(self, agent_response):
        """Process agent response and return user reply."""
        agent_response = f"[Agent] {agent_response}"
        self.messages.append({"role": "user", "content": agent_response})
        self.conversations.append({"agent": agent_response})
        raw_response, user_content = self._infer()
        user_content = f"{user_content}"
        self.messages.append({"role": "assistant", "content": raw_response})
        self.conversations.append({"user": user_content})
        return user_content
       
    
    def _infer(self):
        """Infer user response from LLM with retry mechanism."""
        cur_try = 0
        max_try = 5
        while cur_try < max_try:
            cur_try += 1
            raw_response = llm_inference(
                model=self.model,
                messages=self.messages,
                provider=self.provider,
                api_key=self.api_key,
                base_url=self.base_url
            )
            parse_success, user_content = self._parse_response(raw_response)
            if parse_success:
                break
        return raw_response, user_content
    
    def _parse_response(self, text: str):
        """
        Parse response containing # Thought: and # Reply: sections.
        """
        if "###STOP###" in text:
            return True, "###STOP###"
        
        # Use DOTALL + non-greedy matching for multiline content
        pattern = re.compile(
            r'# Thought:\s*(.*?)\s*# Reply:\s*(.*)',
            re.DOTALL
        )
        match = pattern.search(text)
        if match:
            thought_content = match.group(1).strip()
            reply_content = match.group(2).strip()
            return True, reply_content
        else:
            print(f"Parsed response failed: {text}")
            return False, ""
        
    def get_messages(self):
        """Return a deep copy of messages."""
        return deepcopy(self.messages)