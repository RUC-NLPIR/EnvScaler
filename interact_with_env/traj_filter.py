"""
Track filtering for EnvScaler acquisition
(tracks deemed impossible to complete by the Agent, tracks with incorrect format, tracks that are too short, etc.)
"""
import re
import json

REASONING_FLAG = "think"

print("Reasoning_FLAG:", REASONING_FLAG)

RESPOND_ACTION_NAME = "chat_with_user"

def read_json(file_path):
    """Read JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        


def parse_response(text):
    """
    Parse model raw output text in Prompt (non-FC) mode.
    Extracts (reasoning_content, tool_calls, content).
    Returns: (parse_success, result_dict)
    """
    parse_success = True
    result = {"reasoning_content": None, "tool_calls": None, "content": None}
    text = text.strip()

    # Match think block
    # Requires complete <think>...</think>, otherwise parse fails
    pattern = rf'(?:<|@){re.escape(REASONING_FLAG)}(?:>|@)\s*(.*?)(?:<|@)/{re.escape(REASONING_FLAG)}(?:>|@)'
    think_match = re.search(
        pattern,
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if think_match:
        result["reasoning_content"] = think_match.group(1).strip()
    elif re.search(rf'(?:<|@){re.escape(REASONING_FLAG)}(?:>|@)', text, re.IGNORECASE):
        # Found opening tag but not closed
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

    # Extract content based on presence of think and tool_call
    if think_match and tool_call_content:
        # Both exist -> content between them
        think_end = think_match.end()
        tool_start = tool_call_match.start()
        result["content"] = text[think_end:tool_start].strip()
    elif think_match and not tool_call_content:
        # Only think -> content after think
        think_end = think_match.end()
        result["content"] = text[think_end:].strip()
    elif not think_match and tool_call_content:
        # Only tool_call -> content before tool_call
        tool_start = tool_call_match.start()
        result["content"] = text[:tool_start].strip()
    else:
        # Neither -> entire text is content
        result["content"] = text.strip()

    # Parse tool_call JSON
    if tool_call_content:
        try:
            tool_call_dict = json.loads(tool_call_content)
            # Check required fields
            required_fields = ["name", "arguments"]
            missing = [f for f in required_fields if f not in tool_call_dict]
            if missing:
                parse_success = False
                result["tool_calls"] = [{
                    "error": f"Missing required field(s): {missing}",
                    "raw": tool_call_dict,
                }]
            else:
                result["tool_calls"] = [{"function": tool_call_dict}]
        except json.JSONDecodeError as e:
            parse_success = False
            result["tool_calls"] = [{
                "error": f"Failed to parse tool_call JSON: {e}",
                "raw": tool_call_content,
            }]

    return parse_success, result

def parse_action(struct_response: dict):
    """Parse struct_response to action. Returns: (success, action_dict or None)"""
    if (
        "tool_calls" in struct_response
        and struct_response["tool_calls"] is not None
        and len(struct_response["tool_calls"]) > 0
        and struct_response["tool_calls"][0]["function"] is not None
    ):
        # Parse first tool call
        tool_call = struct_response["tool_calls"][0]
        name = tool_call["function"].get("name")
        kwargs = tool_call["function"]["arguments"]
        if isinstance(kwargs, str):
            try:
                kwargs = kwargs.strip()
                kwargs = json.loads(tool_call["function"]["arguments"])
            except Exception as e:
                kwargs = None
                print(f"Response to Action Failed to parse tool call arguments: {tool_call['function']['arguments']}, error: {e}")
        success = bool(name and isinstance(kwargs, dict))
        if not success:
            return False, None
        else:
            return True, {"name": name, "kwargs": kwargs}
    else:
        # Direct response (no tool call)
        content = struct_response.get("content")
        success = bool(content and isinstance(content, str) and content.strip())
        if not success:
            print(f"struct_response to Action Failed to parse content: {content}, error: content is not a string or is empty")
            return False, None
        else:
            return True, {"name": RESPOND_ACTION_NAME, "kwargs": {"content": content}}

def check_response_format(traj_item):
    """Check if all assistant messages have valid format."""
    for message in traj_item['messages']:
        if message['role'] == 'assistant':
            content = message['content']
            parse_response_success, parse_result = parse_response(content)
            if not parse_response_success:
                return False
            elif parse_result['reasoning_content'] is None:
                return False
            elif not parse_result['tool_calls'] and not parse_result['content']:
                return False
            else:
                parse_action_success, action = parse_action(parse_result)
                if not parse_action_success:
                    return False
    return True

    
    
def check_turn(traj_item, max_action, min_action):
    """Filter trajectories with too few or too many actions."""
    action_num = len([1 for message in traj_item['messages'] if message['role'] == 'assistant'])
    return True if min_action <= action_num  <= max_action else False


def check_non_conv_unfinish_traj(traj_item):
    """Check if non-conversation trajectory is completed (must contain 'Task Completed')."""
    last_agent_response = [message for message in traj_item['messages'] if message['role'] == 'assistant'][-1]['content']
    if f"<{REASONING_FLAG}>" not in last_agent_response:
        return False
    last_agent_response = last_agent_response.split(f"</{REASONING_FLAG}>")[1].strip()
    if 'Task Completed' not in last_agent_response:
        # print(traj_item["actions"][-1]['action'])
        return False
    return True

def check_conv_unfish_traj(traj_item):
    """Check if conversation trajectory is completed (must contain '###STOP###')."""
    # if traj_item["messages"][-1]['role'] != "assistant" or "###STOP###" not in traj_item["user_agent_messages"][-1]['content']:
    if traj_item["final_observation"]["type"] != "user" or "###STOP###" not in traj_item["final_observation"]['content']:
        # print(traj_item["user_agent_messages"][-1])
        return False
    return True


def check_return_failed_func_call(traj_item):
    """Count failed Finish function calls (not implemented)."""
    failed_func_num = 0
    for action in traj_item["actions"]:
        if action['action']['name'] == "Finish" and "success" not in action['action']['arguments']['result']:
            failed_func_num += 1


            
def check_non_conv_data(data, max_action = 29, min_action = 3):
    """Filter non-conversation trajectory data through multiple checks."""
    print("total num:", len(data))
    data_temp = [item for item in data if check_response_format(item)]
    print("after check_response_format:", len(data_temp))
    data_temp = [item for item in data_temp if check_turn(item, max_action, min_action)]
    print("after check_turn:", len(data_temp), "max_action:", max_action, "min_action:", min_action)
    data_temp = [item for item in data_temp if check_non_conv_unfinish_traj(item)]
    print("after check_unfinish_traj:", len(data_temp))
    return data_temp
    
def check_conv_data(data, max_action = 39, min_action = 5):
    """Filter conversation trajectory data through multiple checks."""
    print("total num:", len(data))
    data_temp = [item for item in data if check_response_format(item)]
    print("after check_response_format:", len(data_temp))
    data_temp = [item for item in data_temp if check_turn(item, max_action, min_action)]
    print("after check_turn:", len(data_temp), "max_action:", max_action, "min_action:", min_action)
    data_temp = [item for item in data_temp if check_conv_unfish_traj(item)]
    print("after check_unfinish_traj:", len(data_temp))
    return data_temp
    
def analysis_traj(data):
    """Analyze trajectory statistics (action count distribution)."""
    # Count actions per trajectory
    action_num_list = []
    for traj_item in data:
        action_num = len([1 for message in traj_item['messages'] if message['role'] == 'assistant'])
        action_num_list.append(action_num)
    action_num_list.sort()
    action_num_dict = {}
    for action_num in action_num_list:
        if action_num not in action_num_dict:
            action_num_dict[action_num] = 0
        action_num_dict[action_num] += 1
    # print("action_num_list:", action_num_list)
    print("action_num_dict:", action_num_dict)
    print("action_num_list median:", action_num_list[len(action_num_list)//2])
    print("action_num_list mean:", round(sum(action_num_list) / len(action_num_list), 2))

    


if __name__ == "__main__":
    # Non Conversation Traj Filter
    read_file_path = 'result/envscaler_non_conversation_sft/your_non_conversation_traj.json'
    save_file_path = read_file_path.replace('.json', '_filtered.json')
    data = read_json(read_file_path)
    data_filtered = check_non_conv_data(data)
    analysis_traj(data_filtered)
    save_json(save_file_path, data_filtered) 
    
    # Conversation Traj Filter
    read_file_path = 'result/envscaler_conversation_sft/your_conversation_traj.json'
    save_file_path = read_file_path.replace('.json', '_filtered.json')
    data = read_json(read_file_path)
    data_filtered = check_conv_data(data)
    analysis_traj(data_filtered)
    save_json(save_file_path, data_filtered) 