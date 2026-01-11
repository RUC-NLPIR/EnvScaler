"""
step2: After generating test initialization state, perform iterative tool calls and checks for each environment to generate test results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from copy import deepcopy

from utils.process_file import read_file, save_file
from check_util.auto_env import build_env_from_str, get_state_diff
from check_util.func_call_agent import FuncCallAgent
from check_util.check_agent import CheckAgent


def build_func_test_cases(steps_log):
    """
    Convert steps_log to function-grouped log structure with three-state results (pass/warning/fail).
    Preserves Agent Thought and Check Agent reasoning.
    """
    def normalize_status(case):
        """Normalize status from status field or legacy passed field."""
        status = (case.get("status") or "").strip().lower()
        if status in ("pass", "warning", "fail"):
            return status
        # Fallback to legacy passed field
        if case.get("passed") is True:
            return "pass"
        if case.get("passed") is False:
            return "fail"
        # Default to fail
        return "fail"

    log_data = {
        "func_test_cases": {
            "summary": {
                "total_count": 0,
                "pass_count": 0,
                "warning_count": 0,
                "fail_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_pass_count": 0,
                "positive_warning_count": 0,
                "positive_fail_count": 0,
                "negative_pass_count": 0,
                "negative_warning_count": 0,
                "negative_fail_count": 0
            },
            "details": {}
        }
    }

    summary_all = log_data["func_test_cases"]["summary"]
    details = log_data["func_test_cases"]["details"]

    for case in steps_log:
        func_name = case.get("tool_name", "UNKNOWN")
        thought = case.get("thought", "")
        case_type = (case.get("case_type") or "").strip().lower()
        status = normalize_status(case)
        passed_bool = (status == "pass")
        check_reason = None

        # Extract reasoning from Check Agent response (supports analysis or reason fields)
        if isinstance(case.get("check_result"), dict):
            check_reason = case["check_result"].get("analysis") \
                           or case["check_result"].get("reason") \
                           or None

        # Update global summary
        summary_all["total_count"] += 1
        if status == "pass":
            summary_all["pass_count"] += 1
        elif status == "warning":
            summary_all["warning_count"] += 1
        else:
            summary_all["fail_count"] += 1

        if case_type == "positive":
            summary_all["positive_count"] += 1
            if status == "pass":
                summary_all["positive_pass_count"] += 1
            elif status == "warning":
                summary_all["positive_warning_count"] += 1
            else:
                summary_all["positive_fail_count"] += 1
        elif case_type == "negative":
            summary_all["negative_count"] += 1
            if status == "pass":
                summary_all["negative_pass_count"] += 1
            elif status == "warning":
                summary_all["negative_warning_count"] += 1
            else:
                summary_all["negative_fail_count"] += 1

        # Initialize function record if not exists
        if func_name not in details:
            details[func_name] = {
                "summary": {
                    "total_count": 0,
                    "pass_count": 0,
                    "warning_count": 0,
                    "fail_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "positive_pass_count": 0,
                    "positive_warning_count": 0,
                    "positive_fail_count": 0,
                    "negative_pass_count": 0,
                    "negative_warning_count": 0,
                    "negative_fail_count": 0
                },
                "cases": []
            }

        # Update function-level summary
        summary_func = details[func_name]["summary"]
        summary_func["total_count"] += 1
        if status == "pass":
            summary_func["pass_count"] += 1
        elif status == "warning":
            summary_func["warning_count"] += 1
        else:
            summary_func["fail_count"] += 1

        if case_type == "positive":
            summary_func["positive_count"] += 1
            if status == "pass":
                summary_func["positive_pass_count"] += 1
            elif status == "warning":
                summary_func["positive_warning_count"] += 1
            else:
                summary_func["positive_fail_count"] += 1
        elif case_type == "negative":
            summary_func["negative_count"] += 1
            if status == "pass":
                summary_func["negative_pass_count"] += 1
            elif status == "warning":
                summary_func["negative_warning_count"] += 1
            else:
                summary_func["negative_fail_count"] += 1

        # Build case record with key fields plus thought and reasoning
        case_record = {
            "step": case.get("step"),
            "case_type": case_type,
            "status": status,
            "passed": passed_bool,  # Legacy boolean field (warning will be False)
            "thought": thought,
            "parameters": case.get("parameters", {}),
            "state_before_call": case.get("state_before_call", {}),
            "state_after_call": case.get("state_after_call", {}),
            "state_diff": case.get("state_diff", {}),
            "observation": case.get("observation", {}),
            "check_result": case.get("check_result", {}),
            "check_reason": check_reason
        }

        details[func_name]["cases"].append(case_record)

    return log_data

def run_step(step_idx, env, func_call_agent, check_agent):
    """Run single test step and return step record dictionary."""
    print("-" * 30)
    print(f"Step {step_idx}")

    # Get current state before call
    state_before_call = deepcopy(env.get_state_info())

    # Generate function call request using rollout agent
    func_call_request = func_call_agent.gen_func_call_request(
        current_state=state_before_call
    )
    func_name = func_call_request['tool_name']
    func_params = func_call_request['parameters']
    case_type = func_call_request.get('case_type')  # Record positive/negative case type

    # Execute function call in environment
    observation, reward, terminated, truncated, info = env.env_step(
        action={"name": func_name, "params": func_params}
    )

    # Get state after call and compute differences
    state_after_call = deepcopy(env.get_state_info())
    state_diff = deepcopy(get_state_diff(state_before_call, state_after_call))

    # White-box checking using check agent
    check_result = deepcopy(check_agent.check_func_call(
        func_name=func_name,
        state_before_call=state_before_call,
        func_params=func_params,
        func_return=observation,
        state_after_call=state_after_call,
        state_diff=state_diff
    ))
    
    if check_result['result']:
        check_status = check_result['result'].lower()
    else:
        check_status = "fail"
    # Update statistics with case type (positive/negative)
    func_call_agent.update_stats(func_name, case_type=case_type, check_result=check_status)

    print(f"Check Result: {check_status}")

    # Build step log record
    step_log = {
        "step": step_idx,
        "tool_name": func_name,
        "case_type": case_type,  # Record positive/negative case type in log
        "parameters": func_params,
        "state_before_call": state_before_call,
        "state_after_call": state_after_call,
        "state_diff": state_diff,
        "observation": observation,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "check_result": check_result,
        "status": check_status,
        "stats_summary": func_call_agent.get_stats_table_str()
    }

    return step_log



def process_item(env_item, model, temperature, max_steps):
    """Process environment item: build env, run test steps, and generate test results."""
    # Build environment
    try:
        env = build_env_from_str(
            env_str=env_item['env_class_code'],
            class_name=env_item['env_class_name'],
            max_steps=10000,
        )
        # Use first init_config
        init_config = env_item['init_config_list'][0]
        env.env_init(init_config=init_config)
    except Exception as e:
        print("build env error:", e)
        return env_item

    # Initialize agents
    func_call_agent = FuncCallAgent(model=model, temperature=temperature, env_item=env_item)
    check_agent = CheckAgent(model=model, temperature=0, env_item=env_item)

    # Run test steps
    steps_log = []
    for i in range(max_steps):
        print(i)
        try:
            step_log = run_step(i, env, func_call_agent, check_agent)
            steps_log.append(step_log)
        except Exception as e:
            print("run step error:", e)
            continue

    # Convert to function-grouped structure
    final_log = build_func_test_cases(steps_log)
    new_item = deepcopy(env_item)
    new_item["func_test_result"] = final_log
    return new_item

def main(read_file_path, save_file_path, model, temperature, max_steps, num_workers, chunk_size):
    """Process all environment items sequentially and save results periodically."""
    raw_data = read_file(read_file_path)
    new_data = []
    for item in tqdm(raw_data, desc="Rollout Check"):
        new_item = process_item(item, model, temperature, max_steps)
        new_data.append(new_item)
        # Save after each item
        if len(new_data) % 1 == 0:
            save_file(save_file_path, new_data)
    save_file(save_file_path, new_data)


# import multiprocessing as mp
# from functools import partial

# # define a wrapper function, to pass parameters to multiprocessing
# def worker(item, model, temperature, max_steps):
#     try:
#         return process_item(item, model, temperature, max_steps)
#     except Exception as e:
#         print("process item error:", e)
#         return item

# # main function (multiprocessing version)
# def main(read_file_path, save_file_path, model, temperature, max_steps, num_workers, chunk_size):
#     raw_data = read_file(read_file_path)
#     total_items = len(raw_data)

#     worker_func = partial(worker, model=model, temperature=temperature, max_steps=max_steps)

#     results = []
#     processed_count = 0  # record the number of completed items

#     # use process pool
#     with mp.Pool(processes=num_workers) as pool:
#         # tqdm show total progress
#         it = pool.imap(worker_func, raw_data)

#         # batch processing, save every chunk_size items
#         chunk = []
#         for res in tqdm(it, total=total_items, desc="Rollout Check"):
#             chunk.append(res)
#             processed_count += 1

#             if len(chunk) >= chunk_size:
#                 results.extend(chunk)
#                 save_file(save_file_path, results)
#                 chunk.clear()  # clear the batch
#                 print(f"saved progress: {processed_count}/{total_items}")

#         # if there are remaining items not saved
#         if chunk:
#             results.extend(chunk)
#             save_file(save_file_path, results)
#             print(f"saved final progress: {processed_count}/{total_items}")

#     print(f"task completed, file saved to {save_file_path}")
    
if __name__ == "__main__":
    model = "gpt-4.1-mini"
    read_file_path = "stage3_check_env/temp_result/step1_gen_test_init_config.json"
    save_file_path = "stage3_check_env/temp_result/step2_roll_check.json"
    temperature = 0.5
    max_steps = 30  # Maximum test loop count per environment
    num_workers = 1  # Number of workers for multiprocessing
    chunk_size = 1  # Save after every chunk_size environments
    main(read_file_path, save_file_path, model, temperature, max_steps, num_workers, chunk_size)