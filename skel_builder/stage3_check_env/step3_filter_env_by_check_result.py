"""
step3: Filter environments based on check results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.process_file import read_file, save_file


def get_not_fail_accs(data):
    """Get not_fail accuracy (pass + warning rate) for each environment."""
    not_fail_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        not_fail_acc = round((func_test_result_summary["pass_count"]+ func_test_result_summary["warning_count"]) / func_test_result_summary["total_count"], 2)
        not_fail_acc_list.append(not_fail_acc)
    return not_fail_acc_list

def get_postive_not_fail_accs(data):
    """Get positive not_fail accuracy (positive pass + warning rate) for each environment."""
    postive_not_fail_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        postive_not_fail_acc = round((func_test_result_summary["positive_pass_count"]+ func_test_result_summary["positive_warning_count"]) / func_test_result_summary["positive_count"], 2)
        postive_not_fail_acc_list.append(postive_not_fail_acc)
    return postive_not_fail_acc_list

def get_pass_accs(data):
    """Get pass accuracy (pass rate only) for each environment."""
    pass_acc_list = []
    for item in data:
        func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
        pass_acc = round(func_test_result_summary["pass_count"] / func_test_result_summary["total_count"], 2)
        pass_acc_list.append(pass_acc)
    return pass_acc_list

def select_env(data, select_field, threshold):
    """Filter environments based on selected field and threshold."""
    new_data = []
    if select_field == "not_fail":
        # Filter by pass + warning rate
        for item in data:
            func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
            not_fail_acc = (func_test_result_summary["pass_count"]+ func_test_result_summary["warning_count"]) / func_test_result_summary["total_count"]
            if not_fail_acc >= threshold:
                new_data.append(item)
        return new_data
    elif select_field == "pass":
        # Filter by pass rate only
        for item in data:
            func_test_result_summary = item["func_test_result"]['func_test_cases']['summary']
            pass_acc = func_test_result_summary["pass_count"] / func_test_result_summary["total_count"]
            if pass_acc >= threshold:
                new_data.append(item)
        return new_data
    else:
        raise ValueError("unknown select_field: {}".format(select_field))
    
def brief_metadata(data):
    """Simplify environment metadata by removing detailed fields."""
    for item in data:
        if "env_func_details" in item:
            del item["env_func_details"]
        if "func_test_result" in item:
            del item["func_test_result"]['func_test_cases']['details']
    return data

def process_env_metadata(data):
    """Process data: reassign env_id and extract key metadata fields."""
    new_data = {}
    env_count = 0
    for item in data:
        # Reassign env_id
        env_count += 1
        env_id = f"env_{env_count}"
        # Extract key environment metadata fields
        new_item = {
            "env_id": env_id, 
            "environment_summary": item["environment_summary"], 
            "environment_introduction": item["environment_introduction"], 
            "state_space_definition": item["state_space_definition"], 
            "constraints_rules": item["constraints_rules"], 
            "operation_list": item["operation_list"], 
            "env_class_name": item["env_class_name"], 
            "env_class_code": item["env_class_code"], 
            "env_class_def": item["env_class_def"],
            "env_structure": item["env_structure"], 
            "tools": item["tools"]
        }
        new_data[env_id] = new_item
    return new_data
        
    


if __name__ == "__main__":
    # Filter items with func_test_result
    data = read_file('stage3_check_env/temp_result/step2_roll_check.json')
    data = [item for item in data if "func_test_result" in item]
    print(len(data))

    # not_fail_acc_list = get_not_fail_accs(data)
    # count = ratio_by_auto_threshold(not_fail_acc_list, 0.05)
    # print(count)
    # postive_not_fail_acc_list = get_postive_not_fail_accs(data)
    # count = ratio_by_auto_threshold(postive_not_fail_acc_list, 0.05)
    # print(count)
    # pass_acc_list = get_pass_accs(data)
    # count = ratio_by_auto_threshold(pass_acc_list, 0.05)
    # print(count)

    # Select environments by not_fail accuracy threshold
    threshold = 0.85
    selected_env_data = select_env(data, "not_fail", threshold)
    print(len(selected_env_data))
    save_file('stage3_check_env/temp_result/step3_selected_env_data.json', selected_env_data)
    # Process metadata and save filtered results
    env_metadata = process_env_metadata(selected_env_data)
    save_file('stage3_check_env/final_result/filtered_env_metadata.json', brief_metadata(env_metadata))