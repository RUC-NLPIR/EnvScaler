"""
Utility functions for parsing model response.
"""
import re
import json
from ..tau_bench_types import Action, RESPOND_ACTION_NAME


def parse_response(text):
    """
    Parse (reasoning_content, tool_calls, content) from model raw output text in Prompt (non-FC) mode.
    """
    parse_success = True  # Default success, only False on structural errors
    result = {"reasoning_content": None, "tool_calls": None, "content": None}
    text = text.strip()

    # Match think block
    # Require <think>...</think> to be complete, otherwise parse_fail
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
        # Check for unclosed tool_call tag (e.g. only <tool_call> without </tool_call>)
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
            # Check fields
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
    """Parse struct_response to action"""
    if (
        "tool_calls" in struct_response
        and struct_response["tool_calls"] is not None
        and len(struct_response["tool_calls"]) > 0
        and struct_response["tool_calls"][0]["function"] is not None
    ):
        # Parse the first tool call
        tool_call = struct_response["tool_calls"][0]
        name = tool_call["function"].get("name")
        kwargs = tool_call["function"]["arguments"]
        if isinstance(kwargs, str):
            try:
                kwargs = kwargs.strip()
                kwargs = json.loads(tool_call["function"]["arguments"]) if tool_call["function"]["arguments"] else {}
            except Exception as e:
                kwargs = None
                print(f"Response to Action Failed to parse tool call arguments: {tool_call['function']['arguments']}, error: {e}")
        success = bool(name and isinstance(kwargs, dict))
        if not success:
            return False, None
        else:
            return True, Action(
                name=name,  # Tool function name
                kwargs=kwargs,  # Tool arguments
            )
    else:
        # Direct reply
        content = struct_response.get("content")
        success = bool(content and isinstance(content, str) and content.strip())
        if not success:
            print(f"struct_response to Action Failed to parse content: {content}, error: content is not a string or is empty")
            return False, None
        else:
            return True, Action(
                name=RESPOND_ACTION_NAME, 
                kwargs={"content": content}
            )
