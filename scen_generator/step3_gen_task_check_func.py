"""
Step 3: Generate check functions for tasks to verify task completion.
"""
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.process_file import read_file, save_file
from task_check_util.gen_checklist import gen_checklist
from task_check_util.gen_check_func import gen_check_func


def process_single_task(model, task_item, env_items):
    """Process a single task to generate checklist and check functions."""
    env_id = task_item["env_id"]
    env_item = env_items[env_id]
    assert env_item["env_class_name"] == task_item["env_class_name"]

    env_introduction = env_item["environment_introduction"]
    new_task_item = deepcopy(task_item)
    task = task_item["task"]
    init_config = task_item["init_config"]

    # Generate checklist
    check_list = gen_checklist(model, task=task_item["task"])
    # print(f"check_list: {check_list}")
    new_task_item["checklist"] = deepcopy(check_list)
    checklist_with_func = []

    # Generate check function for each check item
    for check_item in check_list:
        check_func = gen_check_func(
            model=model,
            init_config=init_config,
            task=task,
            env_introduction=env_introduction,
            check_item=check_item
        )
        checklist_with_func.append({"check_item": check_item, "check_func": check_func})
    new_task_item["checklist_with_func"] = checklist_with_func

    return new_task_item

    
def main(model, task_items_path, env_items, save_file_path, num_workers):
    """Main function: process tasks sequentially."""
    task_items = read_file(task_items_path)
    results = []
    for task_item in tqdm(task_items):
        new_task_item = process_single_task(model, task_item, env_items)
        results.append(new_task_item)
        if len(results) % 5 == 0:  # Periodic save
            save_file(save_file_path, results)
    print(f"Save to file: {save_file_path}")
    save_file(save_file_path, results)
    # Save to final_result directory as final scenario data
    save_file("final_result/env_scenario.json", results)


def main_threaded(model, task_items_path, env_items, save_file_path, num_workers):
    """Main function: process tasks in parallel using thread pool, preserving original order."""
    # Store results with original index as key
    result_dict = {}
    task_items = read_file(task_items_path)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map future to original index
        future_to_index = {
            executor.submit(process_single_task, model, task_item, env_items): idx
            for idx, task_item in enumerate(task_items)
        }
        pbar = tqdm(total=len(task_items))
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result_dict[idx] = future.result()
            pbar.update(1)

            # Save every 5 items (sorted by original index)
            if len(result_dict) % 5 == 0:
                sorted_results = [result_dict[i] for i in sorted(result_dict.keys())]
                save_file(save_file_path, sorted_results)
        pbar.close()

    # Final save with complete sorting
    sorted_results = [result_dict[i] for i in sorted(result_dict.keys())]
    print(f"Save to file: {save_file_path}")
    save_file(save_file_path, sorted_results)
    # Save to final_result directory as final scenario data
    save_file("final_result/env_scenario.json", sorted_results)


if __name__ == "__main__":
    model = "gpt-4.1-mini"
    task_items_path = "temp_result/step2_gen_task.json"
    env_items = read_file("your_path/filtered_env_metadata.json")
    save_file_path = "temp_result/step3_gen_task_check_func.json"
    num_workers = 3
    main_threaded(model, task_items_path, env_items, save_file_path, num_workers)