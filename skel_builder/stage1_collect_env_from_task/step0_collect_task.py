"""
step0: collect tasks from existing instruction-following datasets (ToolAce, API-Bank).
"""
# add skel_builder directory to sys.path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
from utils.process_file import read_file, save_file


def contains_non_english(s):
    """Check if task contains non-English characters."""
    # English letters + numbers + common English punctuation + whitespace
    allowed_pattern = r'^[A-Za-z0-9\s\.\,\?\!\;\:\'\"\(\)\[\]\{\}\-\_\*/\\@#\$%\^&\+\=<>\|~`]*$'
    return not re.match(allowed_pattern, s)


def is_multimodal_task(task_str):
    """Check if task string contains multimodal keywords."""
    keywords = ["image", "photo", "picture", "video", "audio", "sound", "speech", "clip"]
    return any(kw.lower() in task_str.lower() for kw in keywords)



def extract_task(item, dataset_name):
    """Extract task from a single sample based on dataset format."""
    if dataset_name == 'toolace':
        assert item['conversations'][0]['from'] == 'user'
        task = item['conversations'][0]['value']
    elif dataset_name == "api-bank":
        task = item["input"].split("User:")[1].split("\nGenerate API Request: ")[0].split("\nAPI-Request: ")[0].split("TIME: ")[0].strip()
        if '\nAI: ' in task:
            task = task.split('\nAI: ')[0]
    # Filter non-English tasks
    if contains_non_english(task):
        return False, None
    # Filter multimodal tasks
    if is_multimodal_task(task):
        return False, None
    # Filter tasks with special characters/keywords
    if "Role definition:" in task or 'USD' in task or 'ETH' in task or 'Bitcoin' in task or 'Ethereum' in task or '@' in task or '.com' in task or 'http:' in task or 'https:' in task:
        return False, None

    return True, task


def extract_api_bank():
    """Extract tasks from API-Bank dataset."""
    data = []
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv1-train.json"))
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv2-train.json"))
    data.extend(read_file("stage1_collect_env_from_task/source_data/api-bank/lv3-train.json"))
    tasks = []
    for item in data:
        success, task = extract_task(item, dataset_name='api-bank')
        if success:
            tasks.append(task)
    tasks = list(set(tasks))
    new_data = [{"task": task, "task_from": "api-bank"} for task in tasks]
    return new_data


def extract_toolace():
    """Extract tasks from ToolAce dataset."""
    data = read_file('stage1_collect_env_from_task/source_data/toolace/data.json')
    tasks = []
    for item in data:
        success, task = extract_task(item, dataset_name='toolace')
        if success:
            tasks.append(task)
    tasks = list(set(tasks))
    new_data = [{"task": task, "task_from": "toolace"} for task in tasks]
    return new_data
    

if __name__ == "__main__":
    task_data_1 = extract_api_bank()
    print(len(task_data_1))
    task_data_2 = extract_toolace()
    print(len(task_data_2))
    task_data = task_data_1 + task_data_2
    print(len(task_data))
    save_file("stage1_collect_env_from_task/temp_result/step0_source_tasks.json", task_data)