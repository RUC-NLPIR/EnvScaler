"""
Step 1: Generate initial state configurations for each environment for scenario instantiation.
"""
import json
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file
from utils.auto_env import build_env_from_str


# Prompt template for generating initialization configs
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
    """Parse LLM response to extract analysis and init config JSON."""
    if "# Analysis" in response and "# Init Config" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Init Config")[0].strip()
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
    """Generate initialization config for an environment using LLM."""
    cur_try = 0
    max_try = 3
    init_config = None
    input_content = input_template.format(env_class_code=env_class_code,
                                          all_containers=all_containers)
    # print(input_content)
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


def test_env_config(env_item, init_config):
    """Test if the init config can successfully initialize the environment."""
    env = build_env_from_str(
        env_str=env_item['env_class_code'],
        class_name=env_item['env_class_name'],
        max_steps=10000,
    )
    try:
        env.env_init(init_config=init_config)
    except Exception as e:
        print("build env error:", e)
        return False
    return True


def process_env_item(env_item, gen_config_num, model, temperature, init_config_item = None):
    """Generate multiple init configs for a single environment."""
    env_class_code = env_item["env_class_code"]
    all_containers = {
        k: v
        for k, v in env_item["env_structure"]["states"].items()
        if k != "init_config"
    }
    if init_config_item and "init_config_list" in init_config_item:
        init_config_list = deepcopy(init_config_item["init_config_list"])
    else:
        init_config_list = []

    gen_config_num = gen_config_num - len(init_config_list)
    print(gen_config_num)
    for i in range(gen_config_num):
        init_config = gen_init_config(env_class_code=env_class_code, all_containers=all_containers, model=model, temperature=temperature)
        if init_config and test_env_config(env_item, init_config):
            init_config_list.append(init_config)
    new_item = {"env_id": env_item["env_id"], "env_class_name": env_item["env_class_name"]}
    new_item["init_config_list"] = init_config_list
    return new_item


# Sequential main
# def main(read_file_path, save_file_path, gen_config_num, model, temperature):
#     raw_data = read_file(read_file_path)
#     if isinstance(raw_data, dict):
#         raw_data = list(raw_data.values())
#     new_data = []
#     for env_item in tqdm(raw_data, desc="Gen init config"):
#         new_item = process_env_item(env_item, gen_config_num, model, temperature)
#         new_data.append(new_item)
#         if len(new_data) % 10 == 0:
#             save_file(save_file_path, new_data)
#     save_file(save_file_path, new_data)


def main_thread(read_file_path, save_file_path, gen_config_num, model, temperature, max_workers):
    """Main function: process environments in parallel using thread pool."""
    raw_data = read_file(read_file_path)
    if isinstance(raw_data, dict):
        raw_data = list(raw_data.values())
    print(f"Total env num: {len(raw_data)}")
    # Store results with original index as key
    result_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_env_item, env_item, gen_config_num, model, temperature): idx
            for idx, env_item in enumerate(raw_data)
        }
        for future in tqdm(as_completed(future_to_index), total=len(raw_data)):
            idx = future_to_index[future]
            result_dict[idx] = future.result()

            # Save every 10 items (sorted by original index)
            if len(result_dict) % 10 == 0:
                sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
                save_file(save_file_path, sorted_data)
    # Final save with complete sorting
    sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, sorted_data)


if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "your_path/filtered_env_metadata.json"  # Path to environment metadata file
    save_file_path = "temp_result/step1_init_env_config.json"
    gen_config_num = 50  # Number of configs to generate per environment (equals number of scenarios)
    temperature = 0.7
    max_workers = 3  # Maximum number of concurrent workers
    main_thread(read_file_path=read_file_path, save_file_path=save_file_path, gen_config_num=gen_config_num, model=model, temperature=temperature, max_workers=max_workers)
    # main(read_file_path=read_file_path, save_file_path=save_file_path, gen_config_num=gen_config_num, model=model, temperature=temperature)