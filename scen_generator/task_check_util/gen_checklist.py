"""
Generate checklist items for task verification.
"""
from typing import Tuple, List
from utils.call_llm import llm_inference


# Prompt template for generating checklists
input_template = \
"""You are a **Quality Checklist Generation Assistant**.  
I will provide you with a **task description**. Your job is to generate a **simple and uniformly phrased checklist**.

**Requirements:**
1. Each checklist item must be **independent** and **not rely** on the results of other items.  
2. Every checklist item must start with the **exact phrase**: **"Has …"** followed by a clear description of the action or field to verify.  
3. Use precise fields and exact values; **avoid vague wording**.  
4. If the task requires checking multiple fields, **split them into separate checklist items**.  
5. List the items in **logical order**, ensuring each is **self-contained**.  
6. **Output format:**
   - Use Markdown list syntax (`- `) for each checklist item  
   - Each item must start with **"Has …"** and be **verifiable with a single boolean expression**  

---

**Example:**  

Task description:
Register a new hospital device with ID DEV-9Z88H, model VNT-900, manufactured by Radiant Health Systems, install it at location LOC-RESP-01 (type: ward), and ensure maintenance schedule MSCH-0101 has a `compliance_status` of `compliant`.

Expected CheckList:
- Has the new device DEV-9Z88H been registered?
- Has the device_id of DEV-9Z88H been set to "DEV-9Z88H"?
- Has the model_number of DEV-9Z88H been set to "VNT-900"?
- Has the manufacturer of DEV-9Z88H been set to "Radiant Health Systems"?
- Has the new location LOC-RESP-01 been created?
- Has the type of location LOC-RESP-01 been set to "ward"?
- Has the compliance_status of maintenance schedule MSCH-0101 been set to "compliant"?

---

Now, generate the checklist for the following task:
{task}

Output Format (strictly follow this):
# Analysis
<Your step-by-step reasoning and analysis process>

# CheckList
- Has ...
- Has ...
- ...
"""

def parse_response(output_text: str) -> Tuple[bool, List[str]]:
    """Parse checklist items from LLM output."""
    if "</think>" in output_text:
        output_text = output_text.split("</think>")[1]

    parse_success = False
    checklist = []

    # Normalize line breaks and split
    lines = output_text.strip().splitlines()
    # Flag to detect we are inside "# CheckList" section
    in_checklist_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# CheckList"):
            in_checklist_section = True
            continue
        if in_checklist_section:
            # Stop parsing if we hit another section header
            if stripped.startswith("# ") and not stripped.startswith("# CheckList"):
                break
            # Checklist item lines must start with "- "
            if stripped.startswith("- "):
                checklist.append(stripped[2:].strip())

    if checklist:
        parse_success = True

    return parse_success, checklist

def gen_checklist(model: str, task: str) -> List[str]:
    """Generate checklist items for the given task using LLM."""
    input_content = input_template.format(task=task)
    messages = [
        {"role": "user", "content": input_content},
    ]
    cur_try = 0
    max_try = 5
    while cur_try < max_try:
        cur_try += 1
        output_text = llm_inference(provider="openai", model=model, messages=messages)
        parse_success, checklist = parse_response(output_text)
        if parse_success:
            break
    return checklist