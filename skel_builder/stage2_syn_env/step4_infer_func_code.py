"""
step4: 
Generate function code for each operation in the operation list.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


system_prompt = "You are a code generation assistant.\nGiven an Agent's environment, including the environment's summary and introduction, the environment's state space definition, the environment's constraint rules, key base class definitions, and the list of operations supported by the environment.\nOperations include two types: one is information querying of the environment, and the other is state modification of the environment.\nGiven one of the operations in the operation list (Target Operation),\n\nYou must:  \n1. In **# Analysis**, reason about:  \n   - What entities/attributes are involved.  \n   - Parameters needed.  \n   - Expected outputs (queries return structured results, state modifications return success messages).  \n   - Error/edge cases (e.g., invalid input, permission denied).  \n   - Does it involve environmental constraints or rules.  \n2. In **# Code**, implement the Python method:  \n   - Method name: `def <operation_name>(self, ...)`.  Note: Cannot be an independent function, but rather a method function within an already implemented environment class.\n   - Add clear type hints.  \n   - Add docstring describing inputs, outputs, constraints.  \n   - **Error handling**: do **not raise exceptions** — return a dict like `{ \"success\": False, \"error\": \"reason\" }`.  \n   - For information-query operations, if successful return `{ \"success\": True, \"data\": <result> }`.  \n   - For state-modifying operations, if successful return `{ \"success\": True, \"message\": \"operation description\" }`. \n\nIn each subsequent round, the input format is:\n### Environment Summary\n<environment_summary_here>\n\n### Environment Introduction\n<environment_introduction_here>\n\n### State Space Definition\n<state_space_definition_here>\n\n### Constraints Rules\n<constraints_rules_here>\n\n### Class Definition\n```python\n<class_definition_here>\n```\n\n### Operation List\n{operation_list}\n\n### Target Operation\n{\n  \"operation_name\": \"<operation_name>\",\n  \"operation_description\": \"<operation_description>\",\n  \"operation_type\": \"<query_or_state_change>\"\n}\n\n\nYour output format must be:\n# Analysis\n[Explain reasoning: inputs, outputs, related entities/attributes, constraints logic, success/failure cases]\n\n# Code\n```python\ndef <operation_name>(self, ...):\n    \"\"\"\n    <docstring explaining inputs, outputs and constraints>\n    \"\"\"\n    # Implementation\n```"

input_case_1 = "Based on the following environment specification, generate the function code for the target operation.\n\n### Environment Summary\nLinux filesystem\n\n### Environment Introduction\nThis environment is a Linux filesystem, where hierarchical directories and files are managed with associated metadata such as permissions, ownership, and timestamps. The `/var/tmp/` directory is commonly used for temporary files that need to persist across reboots, and utilities are available to manipulate and query file properties. This stateful environment is ideal for tasks that require combining filename pattern matching, attribute checking, and file compression operations.\n\n### State Space Definition\n[{'entity': 'Directory', 'attributes': 'directory_path, permissions, owner, group, timestamps (created, modified, accessed)', 'description': 'Represents a directory in the filesystem, including its location in the hierarchy and metadata.'}, {'entity': 'File', 'attributes': 'file_path, name, extension, size, permissions, owner, group, timestamps (created, modified, accessed), parent_directory', 'description': 'Represents a file, with metadata and association to its parent directory.'}]\n\n### Constraints Rules\n['Each file must have a unique path within the filesystem.', 'Permissions and ownership determine the allowed operations on files and directories.', 'Timestamps must be automatically updated according to standard filesystem operations.', 'Directory hierarchy is strictly tree', 'structured (no cycles).', 'File names within the same directory must be unique.']\n\n### Class Definition\n```python\n\nfrom typing import Dict, TypedDict\n\nclass TimestampsInfo(TypedDict):\n    created: float  # Unix timestamp\n    modified: float\n    accessed: float\n\nclass DirectoryInfo(TypedDict):\n    directory_path: str\n    permissions: str\n    owner: str\n    group: str\n    timestamps: TimestampsInfo  # created, modified, accessed\n\nclass FileInfo(TypedDict):\n    file_path: str\n    name: str\n    extension: str\n    size: int  # In bytes\n    permissions: str\n    owner: str\n    group: str\n    timestamps: TimestampsInfo  # created, modified, accessed\n    parent_directory: str\n\nclass LinuxFileSystem:\n    def __init__(self, init_config: dict):\n        \"\"\"\n        Linux filesystem environment simulation.\n        init_config: configuration for initialization (not implemented here).\n        \"\"\"\n\n        # Directories: {directory_path: DirectoryInfo}\n        # Represents all directories in the filesystem with metadata.\n        self.directories: Dict[str, DirectoryInfo] = {}\n\n        # Files: {file_path: FileInfo}\n        # Represents all files in the filesystem with metadata.\n        self.files: Dict[str, FileInfo] = {}\n\n        # --- Constraints (to be enforced in methods, not here) ---\n        # 1. Each file must have a unique path within the filesystem.\n        # 2. Permissions and ownership determine allowed operations on files/directories.\n        # 3. Timestamps (created, modified, accessed) must be updated per FS operations.\n        # 4. Directory hierarchy is strictly tree-structured (no cycles).\n        # 5. File names within the same directory must be unique.\n\n        self.init_config = init_config\n\n```\n\n### Operation List\n['list_files_in_directory', 'filter_files_by_extension', 'get_file_info', 'get_file_last_modified_time', 'check_file_permissions', 'get_directory_info', 'compress_file', 'update_file_timestamp', 'delete_file', 'create_file', 'change_file_permissions', 'change_file_owner']\n\n### Target Operation\n\"operation_name\": \"list_files_in_directory\"\n\"operation_description\": \"Retrieve all files within a specified directory path.\"\n\"operation_type\": \"query\""

output_case_1 = "# Analysis\nThe operation `list_files_in_directory` is an information query operation which retrieves all files that reside within a given directory path.\n\n**Entities/Attributes:**\n- Entity: File\n- Relevant attributes: file_path, parent_directory\n\n**Parameters Needed:**\n- `directory_path` (str): the absolute path to the target directory.\n\n**Operation Logic:**\n- Verify that the given `directory_path` exists in `self.directories`. If not, return failure.\n- For each file in `self.files`, check if the `parent_directory` equals the given `directory_path`.\n- Collect information on such files and return a list (likely their FileInfo dictionaries).\n\n**Constraints:**\n- Ensure that the provided directory exists (directory path is valid).\n- Tree structure of directories is implicit, but not directly relevant for a flat fetch.\n\n**Expected Output:**\n- On success: `{ \"success\": True, \"data\": <list of FileInfo for each file where parent_directory == directory_path> }`\n- On error: If directory does not exist, `{ \"success\": False, \"error\": \"Directory does not exist\" }`.\n\n**Edge/Error Cases:**\n- Directory does not exist.\n- Directory exists but contains no files: return empty list (this is still a successful call).\n\nNo permission checking is required (unless explicitly asked in the operation).\n\n# Code\n```python\ndef list_files_in_directory(self, directory_path: str) -> dict:\n    \"\"\"\n    Retrieve all files (with metadata) within the specified directory path.\n\n    Args:\n        directory_path (str): The absolute path to the directory to query.\n\n    Returns:\n        dict: {\n            \"success\": True,\n            \"data\": List[FileInfo],  # List of matching files' info (may be empty if no files)\n        }\n        or\n        {\n            \"success\": False,\n            \"error\": str  # Description of the error, e.g. directory does not exist\n        }\n\n    Constraints:\n        - The given directory must exist in the filesystem.\n    \"\"\"\n    if directory_path not in self.directories:\n        return { \"success\": False, \"error\": \"Directory does not exist\" }\n\n    result = [\n        file_info for file_info in self.files.values()\n        if file_info[\"parent_directory\"] == directory_path\n    ]\n\n    return { \"success\": True, \"data\": result }\n```"

input_template = \
"""
Based on the following environment specification, generate the function code for the target operation.

### Environment Summary
{env_summary}

### Environment Introduction
{env_intro}

### State Space Definition
{state_space_definition}

### Constraints Rules
{constraints_rules}

### Class Definition
```python
{class_definition}
```

### Operation List
{operation_name_list}

### Target Operation
"operation_name": "{operation_name}"
"operation_description": "{operation_description}"
"operation_type": "{operation_type}"
"""

def parse_response(response):
    """Parse the response from the LLM"""
    if "# Analysis" in response and "# Code" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Code")[0].strip()
            code = response.split("# Code")[1].strip().lstrip("```python").rstrip("```").strip()
            return analysis, code
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "Error parsing response: {e}"+ response, None
    else:
        print(f"Error parsing response: {response}")
        return "Error parsing response: "+ response, None
    
def construct_messages(env_item, operation_item):
    """Construct the messages for the LLM"""
    # env info
    env_summary = env_item["environment_summary"]
    env_intro = env_item["environment_introduction"]
    state_space_definition = env_item["state_space_definition"]
    constraints_rules = env_item["constraints_rules"]
    class_definition = env_item["class_definition"]
    operation_name_list = [operation["operation_name"] for operation in env_item["operation_list"]]
    # operation info
    operation_name = operation_item["operation_name"]
    operation_description = operation_item["operation_description"]
    operation_type = operation_item["operation_type"]
    # format
    input_content = input_template.format(
        env_summary=env_summary, 
        env_intro=env_intro, 
        state_space_definition=state_space_definition, 
        constraints_rules=constraints_rules, 
        class_definition=class_definition, 
        operation_name_list=operation_name_list, 
        operation_name=operation_name, 
        operation_description=operation_description, 
        operation_type=operation_type)
    # print(f"Input content: \n{input_content}")
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": input_case_1},
        {"role": "assistant", "content": output_case_1},
        {"role": "user", "content": input_content}
    ]
    return messages

def llm_infer(messages, model):
    """LLM inference"""
    cur_try = 0
    max_try = 5
    parse_success = False
    func_code = ""
    while cur_try < max_try:
        response = llm_inference(
            provider="openai",
            model=model,
            messages=messages)
        parse_success, func_code = parse_response(response)
        if parse_success:
            break
        cur_try += 1
    return func_code

def process_env_item_for_demo(env_item, model):
    """Only for demo: Process the environment item for demo"""
    new_env_item = deepcopy(env_item)
    operation_items = deepcopy(env_item["operation_list"])
    # for i in tqdm(range(len(operation_items)), desc="Processing operation"):
    for i in tqdm(range(len(operation_items)), 
                desc='Processing operation', 
                bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} '): # 删掉了前面的 {desc}:
        messages = construct_messages(env_item, operation_items[i])
        func_code = llm_infer(messages, model)
        operation_items[i]["code"] = func_code

        from env_build_demo import pretty_print
        print(operation_items[i]['operation_name'])
        pretty_print(operation_items[i]['code'], style="python")
        
    new_env_item["operation_list"] = operation_items
    return new_env_item

def process_env_item(env_item, model):
    """Process the environment item"""
    new_env_item = deepcopy(env_item)
    operation_items = deepcopy(env_item["operation_list"])
    for i in tqdm(range(len(operation_items)), desc="Processing operation"):
        messages = construct_messages(env_item, operation_items[i])
        func_code = llm_infer(messages, model)
        operation_items[i]["code"] = func_code
    new_env_item["operation_list"] = operation_items
    return new_env_item

# def main(read_file_path,save_file_path, model):
#     raw_data = read_file(read_file_path)[:1]
#     new_data = []
#     for env_item in tqdm(raw_data):
#         new_env_item = process_env_item(env_item, model)
#         new_data.append(new_env_item)
#         if len(new_data)%5==0:
#             save_file(save_file_path, new_data)
#             raise
#     save_file(save_file_path, new_data) 
    


def main(read_file_path, save_file_path, model, max_workers):
    raw_data = read_file(read_file_path)
    # Use a dictionary to save the results, key is the original index
    result_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_env_item, env_item, model): idx
            for idx, env_item in enumerate(raw_data)
        }
        for future in tqdm(as_completed(future_to_index), total=len(raw_data)):
            idx = future_to_index[future]
            result_dict[idx] = future.result()
            # Save every 10 data (sorted by the order of completion)
            if len(result_dict) % 5 == 0:
                sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
                save_file(save_file_path, sorted_data)
    # Sort once when saving the final result
    sorted_data = [result_dict[i] for i in sorted(result_dict.keys())]
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, sorted_data)


if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "stage2_syn_env/temp_result/step3_infer_operation.json"
    save_file_path = "stage2_syn_env/temp_result/step4_infer_func_code.json"
    main(read_file_path, save_file_path, model, max_workers=2)