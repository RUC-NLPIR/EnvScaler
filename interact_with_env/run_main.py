"""
The main program is used for:
1. Interacting with the EnvScaler synthesized environment to acquire trajectories for training
2. Evaluating the performance of TauBench and AceBench.
"""
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.task_solve_agent import TaskSolveAgent

# Environment imports
from envscaler_env import EnvScalerConvRLEnv, EnvScalerNonConvRLEnv, EnvScalerConvSFTEnv, EnvScalerNonConvSFTEnv
from taubench_env import TauBenchRetailEnv, TauBenchAirlineEnv
from bfcl_env import BfclEnv
from acebench_env import AceBenchMultiStepEnv, AceBenchMultiTurnEnv


# Environment class mapping
env_cls_map = {
    "envscaler_conversation_rl": EnvScalerConvRLEnv,
    "envscaler_non_conversation_rl": EnvScalerNonConvRLEnv,
    "envscaler_conversation_sft": EnvScalerConvSFTEnv,
    "envscaler_non_conversation_sft": EnvScalerNonConvSFTEnv,
    "tau_bench_retail": TauBenchRetailEnv,
    "tau_bench_airline": TauBenchAirlineEnv,
    "bfcl": BfclEnv,
    "acebench_multi_step": AceBenchMultiStepEnv,
    "acebench_multi_turn": AceBenchMultiTurnEnv
}

# Maximum agent-environment interaction steps mapping for each environment
max_steps_map = {
    "envscaler_conversation_rl": 40,
    "envscaler_non_conversation_rl": 30,
    "envscaler_conversation_sft": 40,
    "envscaler_non_conversation_sft": 30,
    "tau_bench_retail": 30,
    "tau_bench_airline": 30,
    "bfcl": 40,
    "acebench_multi_step": 20,
    "acebench_multi_turn": 20
}


def get_current_time():
    """Get current time as formatted string."""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def save_json(path, data):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def solve_task(env_name, env_config, agent_model, agent_model_provider, infer_mode, enable_thinking, task_id):
    # Initialize environment
    try:
        env = env_cls_map[env_name](**env_config)
    except Exception as e:
        raise Exception(f"Error in env_cls_map[{env_name}](**env_config): {repr(e)}")

    max_steps = max_steps_map[env_name]

    # Initialize task solving agent
    agent = TaskSolveAgent(
        env_name=env_name,
        env=env,
        model = agent_model,
        provider = agent_model_provider,
        infer_mode=infer_mode,
        temperature=0.7,
        max_steps=max_steps,
        enable_thinking=enable_thinking
    )
    
    # Execute task
    save_data = {}
    result = agent.run(task_index=task_id)
    save_data.update(result)
    return save_data


def solve_task_multiprocess(task_configs, save_file_path, num_workers):
    """
    multi-process (actually thread pool) execution of solve_task
    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                solve_task,
                cfg["env_name"],
                cfg["env_config"],
                cfg["agent_model"],
                cfg["agent_model_provider"],
                cfg["infer_mode"],
                cfg["enable_thinking"],
                cfg["task_id"]
            )
            for cfg in task_configs
        ]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                # get task result
                res = future.result()
            except Exception as e:
                # single task exception, print and skip
                print(f"[WARNING] Task {i} error, skipped: {e}")
                import traceback
                print(traceback.format_exc())
                continue

            results.append(res)

            if len(results) % 5 == 0:
                save_json(save_file_path, results)

    # save final results
    save_json(save_file_path, results)


def solve_task_single_process(task_configs, save_file_path):
    """
    single thread execution of solve_task
    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    results = []
    for i, cfg in enumerate(tqdm(task_configs, total=len(task_configs))):
        res = solve_task(
            cfg["env_name"],
            cfg["env_config"],
            cfg["agent_model"],
            cfg["agent_model_provider"],
            cfg["infer_mode"],
            cfg["enable_thinking"],
            cfg["task_id"]
        )
        results.append(res)
        # save every 10 results
        if len(results) % 10 == 0:
            save_json(save_file_path, results)

    # save final results
    save_json(save_file_path, results)


if __name__ == "__main__":
    # load env
    load_dotenv()

    # setting
    # set your llm as agent_model
    agent_model = "gpt-4.1-mini"
    agent_model_provider = "openai"

    # Enable Thinking Mode (Only applicable to hybrid thinking models that support thinking switching, such as Qwen3-8B; does not work for other models)
    # enable_thinking = False
    enable_thinking = True
    num_workers = 3

    # EnvScaler-NonConversation-SFT-Env (Training Env)
    # No Reward for SFT
    env_name = "envscaler_non_conversation_sft"
    infer_mode = "prompt"
    env_config = {
        "mode": "train",
        "env_items_path": "envscaler_env/data/191_env_metadata.json",
        "task_items_path": "envscaler_env/data/sft_scenario_metadata.json",
    }
    task_ids = [i for i in range(4684)]

    # EnvScaler-Conversation-SFT-Env (Training Env)
    # No Reward for SFT
    # env_name = "envscaler_conversation_sft"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/sft_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    # }
    # task_ids = [i for i in range(4684)]


    # # EnvScaler-NonConversation-RL-Env (Training Env)
    # env_name = "envscaler_non_conversation_rl"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
    # }
    # task_ids = [i for i in range(2250)]


    # # EnvScaler-Conversation-RL-Env (Training Env)
    # env_name = "envscaler_conversation_rl"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    # }
    # task_ids = [i for i in range(2250)]


    # # TauBench retail (Evaluation Env)
    # env_name = "tau_bench_retail"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_ids = [i for i in range(115)]


    # # TauBench airline (Evaluation Env)
    # env_name = "tau_bench_airline"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_ids = [i for i in range(50)]


    # # AceBench multi-step (Evaluation Env)
    # env_name = "acebench_multi_step"
    # infer_mode = "fc"
    # env_config = {"domain": "agent_multi_step", "truncated_steps": 20}
    # task_ids = [f"agent_multi_step_{i}" for i in range(20)]


    # # AceBench multi-turn (Evaluation Env)
    # env_name = "acebench_multi_turn"
    # infer_mode = "fc"
    # env_config = {
    #     "domain": "agent_multi_turn", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_provider": "openai", 
    #     "truncated_steps": 20}
    # task_ids = [f"agent_multi_turn_{i}" for i in range(30)]


    # print settings
    print(f"agent_model: {agent_model}")
    print(f"agent_model_provider: {agent_model_provider}")
    print(f"infer_mode: {infer_mode}")
    print(f"enable_thinking: {enable_thinking}")

    # generate task configs
    task_configs = [
        deepcopy({
            "env_name": env_name,
            "env_config": env_config,
            "agent_model": agent_model,
            "agent_model_provider": agent_model_provider,
            "infer_mode": infer_mode,
            "enable_thinking": enable_thinking,
            "task_id": task_id
        })
        for task_id in task_ids
    ]
    # generate save file path
    if env_name in ["bfcl", "envscaler_non_conversation_rl","envscaler_non_conversation_sft", "acebench_multi_step"]:
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{get_current_time()}.json"
    elif env_name in ["envscaler_conversation_rl","envscaler_conversation_sft", "acebench_multi_turn"]:
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{env_config['user_model']}_{get_current_time()}.json"
    else: # tau bench
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{env_config['user_model']}_{env_config['user_strategy']}_{get_current_time()}.json"
    print("save_file_path:", save_file_path)
    # run task solving
    solve_task_multiprocess(task_configs=task_configs, save_file_path=save_file_path, num_workers=num_workers)
    # solve_task_single_process(task_configs=task_configs, save_file_path=save_file_path)
    print("save_file_path:", save_file_path)