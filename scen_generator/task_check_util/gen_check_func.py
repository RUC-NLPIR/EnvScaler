"""
Generate check functions for task verification.
"""
import json
import re
import ast
from typing import Tuple
from utils.call_llm import llm_inference


# Prompt template for generating check functions
input_template = \
"""You are a Python verification function generation assistant.  
You will be given:
- Environment introduction (env_introduction), describing the overall context and purpose of the system.
- Environment system configuration (initial_state), defining the structure, field names, and value formats of the system's data, and serving as the environment's initial database state before any modifications occur.
- After a series of modifications performed by an Agent to complete a task, the system reaches the final state (`final_state`).  
- A single check item, phrased as "Has ...", indicating the condition that must be verified against the final state.

Your task:  

Generate a Python function that validates whether the `final_state` satisfies the given check item.

---

Rules:
1. Always reference the data structure, field names, and value formats from `initial_state` when writing your verification logic. This ensures the code matches the actual system schema.
2. If the check item involves randomly generated or time-dependent fields (e.g., `user_id`, `create_time`, `update_time`, UUIDs), do not validate against a fixed concrete value.  
   Instead, check that the field exists in `final_state` and has the correct type/format (e.g., string).
3. If the check item describes a non-fixed target value (e.g., "add a remark"), only verify that the field exists and meets basic conditions (e.g., non-empty string, correct data type).
4. If the check item specifies an explicit target value, you must strictly match it (`==`).
5. Use `initial_state` as a reference only when necessary to determine changes — for example, "Has been added" means the entity didn't exist in `initial_state` but exists in `final_state`.
6. The function must implement only the given single check item and return `True` if passed, `False` if failed.
7. The function must not modify any input data and must perform no actions other than verification.
8. Ensure the function signature is `def check_func(final_state)`, and that it only returns `True` or `False`.

---

Provided data:

# environment introduction:
{env_introduction}

# initial_state:
{init_config}

# complete task:
{task}

# Check item to verify:
{check_item}

---

Required output format (strictly follow):

# Analysis
<Your step-by-step reasoning and analysis process>

# Function
```python
def check_func(final_state):
    ...
```"""


def check_function_code(code_str: str) -> bool:
    """Check if code string is a valid Python function definition."""
    try:
        # Parse code with AST to detect syntax
        tree = ast.parse(code_str)
        # Check if there's a function definition at top level
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return True
        return False
    except SyntaxError:
        return False


def parse_check_func(llm_output: str) -> Tuple[bool, str]:
    """Parse Python function code from LLM output string."""
    if '</think>' in llm_output:
        llm_output = llm_output.split('</think>')[1]
    pattern = re.compile(
        r"# Function\s*```python\s*(.*?)\s*```",
        re.DOTALL  # Allow . to match newlines
    )
    
    match = pattern.search(llm_output)
    if match:
        function_code = match.group(1).strip()
        if not check_function_code(function_code):
            return False, ""
        return True, function_code
    else:
        return False, ""

def gen_check_func(model: str, init_config: dict, task: str, env_introduction: str, check_item: str) -> str:
    """Generate check function for the given task using LLM."""
    init_config_str = json.dumps(init_config, indent=4)
    input_content = input_template.format(
        init_config=init_config_str,
        check_item=check_item,
        task=task,
        env_introduction=env_introduction
    )
    messages = [
        {"role": "user", "content": input_content},
    ]
    cur_try = 0
    max_try = 5
    while cur_try < max_try:
        cur_try += 1
        output_text = llm_inference(provider="openai", model=model, messages=messages)
        parse_success, check_func = parse_check_func(output_text)
        if parse_success:
            break
    return check_func
