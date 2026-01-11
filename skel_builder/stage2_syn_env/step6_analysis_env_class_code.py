"""
step6: Analyze environment class code and extract class name, definition, structure, and method details.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from copy import deepcopy

from utils.process_file import read_file, save_file
from stage2_syn_env.analysis_env_src.get_env_class_def import parse_env_class_name, parse_env_class_def
from stage2_syn_env.analysis_env_src.build_env_structure import build_class_tree_form_str
from stage2_syn_env.analysis_env_src.get_func_details_from_src import extract_class_methods_from_source
from stage2_syn_env.analysis_env_src.get_tool_schema import get_tool_info, convert_tool_schema

def process_env_item(env_item):
    """Extract class name, definition, structure, methods, and tools from environment class code."""
    env_class_code = env_item["env_class_code"]
    env_class_name = parse_env_class_name(env_class_code)
    env_class_def = parse_env_class_def(env_class_code, env_class_name)
    env_structure = build_class_tree_form_str(src = env_class_code, class_name = env_class_name)
    env_func_details = extract_class_methods_from_source(source_str=env_class_code, class_name=env_class_name, include_self=False)
    new_item = deepcopy(env_item)
    print("env_class_name: ", env_class_name)
    new_item["env_class_name"] = env_class_name
    new_item["env_class_def"] = env_class_def
    new_item["env_structure"] = env_structure
    new_item["env_func_details"] = env_func_details
    # Convert tools to schema format
    new_item["tools"] = [convert_tool_schema(tool) for tool in get_tool_info(new_item)]
    return new_item

def main(read_file_path, save_file_path):
    """Process all environment items and save analyzed results."""
    raw_data = read_file(read_file_path)
    new_data = []
    for env_item in raw_data:
        new_item = process_env_item(env_item)
        new_data.append(new_item)
    save_file(save_file_path, new_data)
    # Also save to final result location
    print("Save all data to final_result: stage2_syn_env/final_result/env_with_code.json")
    save_file("stage2_syn_env/final_result/env_with_code.json", new_data)
        
if __name__ == "__main__":
    read_file_path ="stage2_syn_env/temp_result/step5_env_class_code.json"
    save_file_path = "stage2_syn_env/temp_result/step6_analysis_env_class_code.json"
    main(read_file_path,save_file_path)