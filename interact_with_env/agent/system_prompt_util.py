import json
from typing import Any, Callable, Dict, List, Type, Optional


# System prompt for non-conversational action agent
non_conversational_system_prompt =\
"""You are a helpful assistant. When given a specific task, your goal is to complete it in an interactive environment by making step-by-step use of available tools. 
- Before completing the task, at each step, select a tool from the tool list and fill in all required parameters, making sure that the values are valid. Avoid making parallel tool calls in one step.
- When you believe the task has been completed, respond only with 'Task Completed' to end the trajectory, without adding any other content or making any tool calls.
- It is recommended to first call query tools to gather sufficient information, then use modification tools to complete the task. Adjust actions promptly based on the feedback from the environment, i.e., the tool results.
"""

# System prompt for conversational action agent
conversational_system_prompt = \
"""You are a helpful assistant. Your goal is to fulfill the user's requests in an interactive environment by step-by-step use of available tools, while proactively communicating with the user when necessary, until the user ends the conversation.
At each step, you will receive either the user's task/reply or the environment's tool call result.
- If you lack essential information to complete the task or perform a tool call, and it cannot be obtained through the existing tool set, actively ask the user for specific details.  
- If you can proceed with the current information, select one tool from the tool set and provide complete, valid parameters. Avoid making parallel tool calls or calling a tool while interacting with the user in one step.
- It is recommended to first call query tools to gather sufficient information, then use modification tools to complete the task. Adjust actions promptly based on the feedback from the environment, i.e., the tool results.
- When you believe the task is completed, clearly inform the user of the result and ask whether there are any new tasks or follow-up requests. 
"""


def merge_tools_into_system_prompt(
    system_prompt: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    In Prompt (non-FC) mode, merge tool information into system prompt (Qwen3 format).
    """
    # If no tools provided, return system prompt directly
    if not tools:
        return system_prompt if system_prompt else ""
    
    # Build output string
    output = []
    
    # Add system prompt if provided
    if system_prompt:
        output.append(system_prompt)
        output.append("\n\n")
    
    # Add tools section header
    output.append("# Tools\n\n")
    output.append("You may call one or more functions to assist with the user query.\n\n")
    output.append("You are provided with function signatures within <tools></tools> XML tags:\n")
    output.append("<tools>")
    
    # Add JSON representation of each tool
    for tool in tools:
        output.append("\n")
        output.append(json.dumps(tool, ensure_ascii=False))
    
    output.append("\n</tools>\n\n")
    
    # Add tool call instructions
    output.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n")
    output.append("<tool_call>\n")
    output.append('{"name": <function-name>, "arguments": <args-json-object>}\n')
    output.append("</tool_call>")
    
    return ''.join(output)