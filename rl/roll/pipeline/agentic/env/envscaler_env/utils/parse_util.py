"""
Utility functions for parsing model response.
"""
import re
import json
from copy import deepcopy

def parse_response(text):
    """
    Parse (reasoning_content, tool_calls, content) from model raw output text in Prompt (non-FC) mode.
    """
    parse_success = True  # Default success, only False on structural errors
    result = {"reasoning_content": None, "tool_calls": None, "content": None}
    text = text.strip()

    # Match think block
    # Require complete <think>...</think> tags, otherwise parse fails
    think_match = re.search(
        r'(?:<|@)think(?:>|@)\s*(.*?)(?:<|@)/think(?:>|@)',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if think_match:
        result["reasoning_content"] = think_match.group(1).strip()
    elif re.search(r'(?:<|@)think(?:>|@)', text, re.IGNORECASE):
        # Opening tag exists but not closed
        parse_success = False
        result["reasoning_content"] = {"error": "Missing </think> or malformed think block"}

    # Match tool_call block
    tool_calls = list(re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL))
    tool_call_content = None

    if tool_calls:
        if len(tool_calls) > 1:
            print("Multiple <tool_call> found, using the first one.")
        tool_call_match = tool_calls[0]
        tool_call_content = tool_call_match.group(1)
    else:
        # Check for unclosed tool_call tag
        if "<tool_call>" in text and "</tool_call>" not in text:
            parse_success = False
            result["tool_calls"] = [{"error": "Unclosed <tool_call> tag"}]

    # Extract content
    if think_match and tool_call_content:
        # Both think and tool_call exist → content between them
        think_end = think_match.end()
        tool_start = tool_call_match.start()
        result["content"] = text[think_end:tool_start].strip()
    elif think_match and not tool_call_content:
        # Only think → content after think
        think_end = think_match.end()
        result["content"] = text[think_end:].strip()
    elif not think_match and tool_call_content:
        # Only tool_call → content before tool_call
        tool_start = tool_call_match.start()
        result["content"] = text[:tool_start].strip()
    else:
        # Neither exists → entire text is content
        result["content"] = text.strip()

    # Parse tool_call JSON
    if tool_call_content:
        try:
            tool_call_dict = json.loads(tool_call_content)
            # Check required fields
            required_fields = ["name", "arguments"]
            missing = [f for f in required_fields if f not in tool_call_dict]
            if missing:
                parse_success = False  # JSON structure error
                result["tool_calls"] = [{
                    "error": f"Missing required field(s): {missing}",
                    "raw": tool_call_dict,
                }]
            else:
                result["tool_calls"] = [{"function": tool_call_dict}]
        except json.JSONDecodeError as e:
            parse_success = False  # JSON parse error
            result["tool_calls"] = [{
                "error": f"Failed to parse tool_call JSON: {e}",
                "raw": tool_call_content,
            }]

    return parse_success, result


def parse_action(struct_response: dict):
    """Parse struct_response into action."""
    try:
        # Case 1: Only tool calls, no text content
        if not struct_response.get("content") and struct_response.get("tool_calls"):
            action = deepcopy(struct_response["tool_calls"][0]['function'])
            if isinstance(action['arguments'], str):
                action['arguments'] = json.loads(action['arguments'])

        # Case 2: Only text content, no tool calls
        elif struct_response.get("content") and not struct_response.get("tool_calls"):
            action = {
                "name": "chat_with_user",
                "arguments": {"content": struct_response.get("content")}
            }

        # Case 3: Both text content and tool calls → ignore text content
        elif struct_response.get("content") and struct_response.get("tool_calls"):
            action = deepcopy(struct_response["tool_calls"][0]['function'])
            if isinstance(action['arguments'], str):
                action['arguments'] = json.loads(action['arguments'])

        # Case 4: Both are empty → error
        else:
            print("[parse_action] No valid content:", struct_response)
            return False, {}

        return True, action

    except Exception as e:
        print(f"[parse_action] Parse error: {e}, raw data: {struct_response}")
        return False, {}