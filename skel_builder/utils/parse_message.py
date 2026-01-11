"""
Parse model output to extract tool calls and reasoning sections.
"""
import re
import json


def parse_tool_call(text):
    """Parse model output to extract think section and tool_call JSON."""
    parse_success = False
    result = {"think": None, "tool_call": None}
    raw_text = text
    text = text.strip()

    # Extract think section (<think> format)
    think_match = re.search(
        r'(?:<|@)think(?:>|@)\s*(.*?)(?:<|@)/think(?:>|@)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if think_match:
        result["think"] = think_match.group(1).strip()
        # Remove think section from original text
        text = text.replace(think_match.group(0), "").strip()

    # Extract tool_call section (may have multiple)
    tool_calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)

    if not tool_calls:
        result["tool_call"] = {
            "error": "No <tool_call> found",
            "raw": text,
        }
        return parse_success, result

    if len(tool_calls) > 1:
        print("Multiple <tool_call> found, only the first will be used.")   
    tool_call = tool_calls[0]

    # Try to parse JSON
    try:
        tool_call_dict = json.loads(tool_call)
    except json.JSONDecodeError as e:
        result["tool_call"] = {
            "error": f"Failed to parse tool_call JSON: {e}",
            "raw": tool_call,
        }
        return parse_success, result

    # Validate required fields
    required_fields = ["name", "arguments"]
    missing = [f for f in required_fields if f not in tool_call_dict]
    if missing:
        print(f"tool_call missing required field(s): {missing}")
        result["tool_call"] = {
            "error": f"Missing required field(s): {missing}",
            "raw": tool_call_dict,
        }
        return parse_success, result

    # Parse successful
    parse_success = True
    result["tool_call"] = tool_call_dict
    return parse_success, result
