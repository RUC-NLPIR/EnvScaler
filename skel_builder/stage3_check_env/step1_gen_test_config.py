"""
step1: Generate test initialization config for each environment to instantiate it. (Unlike ScenGenerator, no task scenario is needed here.)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file



# Prompt template for generating environment initialization config
input_template = \
"""You are an AI assistant.  
You will be given the complete definition of a Python class.  
This class represents an environment state in a specific domain and contains various attributes (such as dictionaries, lists, `TypedDict` objects, dataclasses, etc.) used to manage entities and their relationships within the system.

Based on the class definition, generate a JSON object that can serve directly as the class's initialization configuration (`config`), following these rules:

---

### 1. Structure and Type Matching  
- The JSON must strictly follow the attribute structure and data types required by the class.  
- Field names, nesting levels, and value types must match the class definition exactly.

### 2. Respect Constraints  
- Read the class methods and docstrings to identify constraints (e.g., valid status values, required fields, ID reference rules), ensuring all generated data complies.  
- All references (e.g., `reporter_id`, `location_id`, `disease_name`) must be cross-linked appropriately and valid.  
- Consider cross-entity relationships and constraints (e.g., a product must belong to an existing category).

### 3. Richness of Data  
- Each major dictionary-like attribute should contain multiple entities (recommended at least 3-5 entries) with differentiated content to avoid repetitive templates.  
- Cover the different states and value ranges supported by the class wherever possible.  
- Dates should be distributed over a reasonable time span to provide diversity.  
- Numerical fields (e.g., `case_count`) should vary in range to simulate realistic system data.

### 4. Realistic Simulation of Data  
- Name fields should use natural-language fictional content (e.g., `"Alice Chan"`, `"Central City District"`) rather than mechanical placeholders like `name1` or `user001`.  
- Description fields should be concise, natural, and logically consistent with the domain's context.  
- Date fields must be in ISO format (`YYYY-MM-DD`) or timestamps, with dates reasonably distributed in time.  
- ID fields may mix short codes (e.g., `LOC1`, `REP1`) and UUIDs, but all must be unique.  
- Data must be fictitious and must not contain any real-world personal or sensitive information.

### 5. Output Format  
- Output only the JSON, without any extra explanation.  
- The JSON must be a complete, ready-to-use initialization configuration that can be passed directly to the class constructor as the `config` parameter.

---

### Env Class Definition
```python
{env_class_code}
```

### All Containers
{all_containers}

---

Strictly follow the following output format:

# Analysis
[Your reasoning: what are the containers, what fields they require, what constraints apply, and how you chose the sample data, etc.]

# Init Config
```json
{{
    ...
}}
```"""

def parse_response(response):
    """Parse LLM response to extract analysis and initialization config JSON."""
    if "# Analysis" in response and "# Init Config" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Init Config")[0].strip()
            # Extract JSON from code block
            init_config = response.split("# Init Config")[1].strip().lstrip("```json").rstrip("```")
            init_config = json.loads(init_config, strict=False)
            return analysis, init_config
        except Exception as e:
            print(f"Error parsing response: {e}")
            return response, None
    else:
        print(f"Error parsing response: {response}")
        return response, None
    
def gen_init_config(env_class_code, all_containers, model, temperature):
    """Generate initialization config using LLM, retry up to max_try times if parsing fails."""
    cur_try = 0
    max_try = 3
    init_config = None
    input_content = input_template.format(env_class_code=env_class_code,
                                          all_containers=all_containers)
    # Retry if parsing fails
    while cur_try < max_try:
        response = llm_inference(
            provider="openai",
            model=model,
            messages=[{"role": "user", "content": input_content}],
            temperature=temperature)
        gen_init_config_analysis, init_config = parse_response(response)
        if init_config:
            break
        cur_try += 1
    return init_config
    

def process_env_item(env_item, gen_config_num, model, temperature):
    """Process environment item to generate required number of initialization configs."""
    env_class_code = env_item["env_class_code"]
    # Extract all containers except init_config
    all_containers = {
        k: v
        for k, v in env_item["env_structure"]["states"].items()
        if k != "init_config"
    }
    # Initialize or copy existing init_config_list
    if "init_config_list" not in env_item:
        init_config_list = []  
    else:
        init_config_list = deepcopy(env_item["init_config_list"])
    
    # Generate only the remaining configs needed
    gen_config_num = gen_config_num - len(init_config_list)
    for i in range(gen_config_num):
        init_config = gen_init_config(env_class_code=env_class_code, all_containers=all_containers, model=model, temperature=temperature)
        if init_config:
            init_config_list.append(init_config)
    new_item = deepcopy(env_item)
    new_item["init_config_list"] = init_config_list
    return new_item

# def main(read_file_path, save_file_path, gen_config_num, model, temperature):
#     raw_data = read_file(read_file_path)
#     new_data = []
#     for env_item in tqdm(raw_data, desc="Gen init config"):
#         new_item = process_env_item(env_item, gen_config_num, model, temperature)
#         new_data.append(new_item)
#         if len(new_data) % 1 == 0:
#             save_file(save_file_path, new_data)
#     save_file(save_file_path, new_data)

def main(read_file_path, save_file_path, gen_config_num, model, temperature, max_workers=3):
    """Process all environment items in parallel and save results periodically."""
    raw_data = read_file(read_file_path)
    # Use dictionary to save results with original index as key
    result_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_env_item, env_item, gen_config_num, model, temperature): idx
            for idx, env_item in enumerate(raw_data)
        }
        for future in tqdm(as_completed(future_to_index), total=len(raw_data)):
            idx = future_to_index[future]
            result_dict[idx] = future.result()

            # Save every 3 items (sorted by completed order)
            if len(result_dict) % 3 == 0:
                sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
                save_file(save_file_path, sorted_data)
    # Final save with complete sorting
    sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, sorted_data)



if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "stage2_syn_env/final_result/env_with_code.json"
    save_file_path = "stage3_check_env/temp_result/step1_gen_test_init_config.json"
    gen_config_num = 1
    temperature = 0.5
    main(read_file_path, save_file_path, gen_config_num, model, temperature)
    