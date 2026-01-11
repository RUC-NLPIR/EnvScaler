"""
Function call agent for generating test cases using LLM.
Generates tool invocations with positive/negative test cases based on environment state and testing history.
"""
import re
import json
import ast
from utils.call_llm import llm_inference

# System prompt template for LLM-based test case generation
system_prompt =\
"""You are an experienced testing engineer, performing comprehensive exploratory testing on all tool interfaces (methods) of a simulated environment class.  
Your goal is to verify the behavior of each method under different types of inputs, aiming to uncover potential errors, exceptions, and state inconsistencies.  
In each upcoming testing round, you will generate one tool invocation as a test case. After execution, you will receive the environment’s return information, along with a result evaluation from backend engineers indicating the test case's result (pass, warning, fail).

[Environment Introduction]  
{env_introduction}

[Available Tool Interface List]  
{tool_info}

Testing Strategy:
- Positive case testing: Use valid parameters that comply with interface definitions (normal input).
- Negative case testing: Use invalid parameters (wrong types, non-existent IDs, out-of-range values, etc.) and special parameters (null/empty values, extreme values, special characters, etc.).
- Throughout testing, cover all available tool interfaces, and ensure a balance between the number of tests for each tool.
- It is not necessary to maintain a consistent task goal; you are free to explore various methods and scenarios.

Testing Rules:
- Invoke only one tool interface per round.
- Parameters must be in dictionary structure; parameter keys must be valid, but parameter values can be invalid or boundary inputs for testing.
- Do not call any methods outside of the provided tool interface list.
- Balance breadth (cover all available methods) and depth (multiple input scenarios for each method) during testing.

[Output Format]  
Strictly follow the format below:

# Thought
<Briefly explain why you chose this method and these parameters>

# Selected Function
<Method name>

# Parameters Dictionary
<Parameter dictionary>"""

# Input template for each test round with testing history and current state
input_template = """
[Tool Testing History Summary]
Here is the testing summary for each tool (Tool | Total | Negative | Positive | Passed | Fail Passed):
{test_summary}

[Current Environment State]
{current_state}

[Available Tools Quick Reference]
{tool_brief_info}

Based on the above information and the testing strategy in the system prompt, continue exploratory testing.
- Whenever possible, choose a tool that is untested or have been tested less frequently to ensure balanced testing.
- For the selected tool, whenever possible, choose the less represented side between positive and negative cases to ensure the balance of positives and negatives.

**Note: The number of all tests summed for any tool cannot exceed 10 times in total.**

Stirngly follow the following output format:
# Thought
<your thought and analysis>

# Selected Function
<tool name>

# Parameters Dictionary
{{JSON format parameters dictionary}}

# Case Type
<only 'positive' or 'negative'>
"""




def get_tool_info(env_item):
    """Extract tool information and convert to JSON string."""
    tool_info_list = []
    for func_name, func_detail in env_item["env_func_details"].items():
        tool_info = {
                     "name": func_name,
                     "description": func_detail["doc"],
                     "parameters": {param["name"]: param["type"] for param in func_detail["signature"]["parameters"]}
                    }
        tool_info_list.append(tool_info)
    return json.dumps(tool_info_list, indent=4)

def get_brief_tool_info(env_item):
    """Generate brief tool information string (function name and parameters)."""
    brief_lines = []
    for func_name, func_detail in env_item["env_func_details"].items():
        params = func_detail["signature"]["parameters"]
        param_str = ", ".join(f"{p['name']}: {p['type']}" for p in params)
        brief_lines.append(f"{func_name}({param_str})")
    return "\n".join(brief_lines)



def parse_response(response_text: str):
    """Parse LLM response to extract tool name, parameters, and case type."""

    try:
        # Pattern to extract Thought, Selected Function, Parameters Dictionary, and Case Type
        pattern = (
            r"#\s*Thought\s*(.*?)\s*"
            r"#\s*Selected\s*Function\s*(.*?)\s*"
            r"#\s*Parameters\s*Dictionary\s*(\{[\s\S]*?\})\s*"
            r"#\s*Case\s*Type\s*['\"]?(positive|negative)['\"]?"
        )
        match = re.search(pattern, response_text, re.S | re.I)
        if not match:
            return False, {}

        thought = match.group(1).strip()  # Available if needed
        raw_tool_name = match.group(2).strip()
        params_str = match.group(3).strip()
        case_type = match.group(4).strip().lower()

        # Normalize tool name: take first non-empty line
        tool_name = ""
        for line in raw_tool_name.splitlines():
            if line.strip():
                tool_name = line.strip()
                break
        if not tool_name:
            return False, {}

        # Parse parameters dictionary (try JSON first, fallback to literal_eval)
        try:
            parameters = json.loads(params_str)
        except Exception:
            try:
                parameters = ast.literal_eval(params_str)
            except Exception:
                return False, {}

        if not isinstance(parameters, dict):
            return False, {}

        data = {
            "tool_name": tool_name,
            "parameters": parameters,
            "case_type": case_type,
            # "thought": thought,  # Uncomment if needed
        }
        return True, data

    except Exception as e:
        print(f"parse_response error: {e}")
        return False, {"error": str(e)}




class FuncCallAgent:
    """Agent for generating function call test cases using LLM."""
    
    def __init__(self, model, temperature, env_item):
        """Initialize function call agent with LLM model and environment item."""
        self.model = model
        self.temperature = temperature
        self.env_item = env_item
        self.tool_info = get_tool_info(env_item)
        self.brief_tool_info = get_brief_tool_info(env_item)
        self.tool_names = list(env_item["env_func_details"].keys())
        # Test statistics tracking
        self.stats = {
            tool_name: {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "pass": 0,
                "warning": 0,
                "fail": 0,
            }
            for tool_name in self.tool_names
        }
        self.system_prompt = system_prompt.format(
            env_introduction=env_item["environment_introduction"], 
            tool_info=self.tool_info)
        self.input_template = input_template

    def update_stats(self, tool_name, case_type, check_result):
        """Update test statistics for a tool after test execution."""
        # Normalize case type and check result
        ct = (case_type or "").strip().lower()
        cr = (check_result or "").strip().lower()

        # Initialize if tool name not in stats to avoid KeyError
        if tool_name not in self.stats:
            self.stats[tool_name] = {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "pass": 0,
                "warning": 0,
                "fail": 0,
            }

        self.stats[tool_name]["total"] += 1

        # Count by case type
        if ct == "positive":
            self.stats[tool_name]["positive"] += 1
        elif ct == "negative":
            self.stats[tool_name]["negative"] += 1

        # Count by check result
        if cr == "pass":
            self.stats[tool_name]["pass"] += 1
        elif cr == "warning":
            self.stats[tool_name]["warning"] += 1
        elif cr == "fail":
            self.stats[tool_name]["fail"] += 1
        else:
            print(f"[update_stats] Unknown check_result value: {check_result}")


    def get_stats_table_str(self):
        """Convert statistics to table string format for prompt."""
        lines = ["Tool | Total | Negative | Positive | Pass | Warning | Fail"]

        for tool, d in self.stats.items():
            lines.append(
                f"{tool} | {d.get('total', 0)} | {d.get('negative', 0)} | {d.get('positive', 0)}"
                f" | {d.get('pass', 0)} | {d.get('warning', 0)} | {d.get('fail', 0)}"
            )

        return "\n".join(lines)

    def get_input(self, current_state):
        """Format input message with test summary, current state, and tool info."""
        input_content = input_template.format(
            test_summary=self.get_stats_table_str(),
            current_state=current_state,
            tool_brief_info=self.brief_tool_info)
        return input_content
    
    def gen_func_call_request(self, current_state):
        """Generate function call request using LLM, retry if parsing fails."""
        input_content = self.get_input(current_state)
        input_message = [{'role': 'system', 'content': self.system_prompt}, {"role": "user", "content": input_content}]
        cur_func_call_try = 0
        max_func_call_try = 3
        # Retry if response parsing fails
        while cur_func_call_try < max_func_call_try:
            response = llm_inference(
                provider="openai",
                model=self.model, 
                temperature=self.temperature, 
                messages=input_message)
            parsed_success, func_call_request = parse_response(response)
            if parsed_success:
                break
        return func_call_request 

