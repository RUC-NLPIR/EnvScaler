"""
Environment debugging script.
"""

import os
import json
import time
from dotenv import load_dotenv

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
    """Solve a task using specified environment and agent."""
    # Initialize environment
    try:
        env = env_cls_map[env_name](**env_config)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Error in env_cls_map[{env_name}](**env_config): {repr(e)}")

    max_steps = max_steps_map[env_name]

    # Initialize task solving agent
    agent = TaskSolveAgent(
        env_name=env_name,
        env=env,
        model = agent_model,
        provider = agent_model_provider,
        infer_mode=infer_mode,
        temperature=0.5,
        max_steps=max_steps,
        enable_thinking=enable_thinking
    )
    
    # Execute task
    save_data = {}
    result = agent.run(task_index=task_id)
    save_data.update(result)
    return save_data




if __name__ == "__main__":
    """
    Uncomment to select the environment and task to run.
    """
    # Load environment variables
    load_dotenv()

    # Agent settings
    agent_model = "gpt-4.1"
    agent_model_provider = "openai"
    
    # infer_mode = "prompt"
    infer_mode = "fc"
    
    # Enable Thinking Mode (Only applicable to hybrid thinking models that support thinking switching, such as Qwen3-8B; does not work for other models)
    # enable_thinking = False
    enable_thinking = True

    
    # EnvScaler-NonConversation-RL-Env 
    # task_id 0-2249
    env_name = "envscaler_non_conversation_rl"
    env_config = {
        "mode": "train",
        "env_items_path": "envscaler_env/data/191_env_metadata.json",
        "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
        }
    task_id = 0

    # # EnvScaler-Conversation-RL-Env
    # # task_id 0-2249
    # env_name = "envscaler_conversation_rl"
    # env_config = {
    #     "mode": "train", 
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    # }
    # task_id = 0
    
    # EnvScaler-NonConversation-SFT-Env
    # env_name = "envscaler_non_conversation_sft"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/sft_scenario_metadata.json",
    #     }
    # task_id = 0
    
    # EnvScaler-Conversation-SFT-Env 
    # env_name = "envscaler_conversation_sft"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/sft_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    #     }
    # task_id = 0
    
    # TauBench retail
    # task_id 0-115
    # env_name = "tau_bench_retail"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model" : "gpt-4.1",
    #     # "user_strategy": "llm",
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_id = 0
    
    # TauBench airline
    # task_id 0-49
    # env_name = "tau_bench_airline"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model" : "gpt-4.1",
    #     # "user_strategy": "llm",
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_id = 0
    
    # BFCL (only support prompt mode for now)
    # task_id multi_turn_base_0-multi_turn_base_199
    # env_name = "bfcl"
    # assert infer_mode == "prompt", "BFCL only supports prompt mode for now"
    # env_config = {"mode": "multi_turn_base"}
    # task_id = "multi_turn_base_0" 
    
    # AceBench-Multi-Step (only FC support for now)
    # task_id agent_multi_step_0-agent_multi_step_19
    # env_name = "acebench_multi_step"
    # env_config = {"domain": "agent_multi_step", "truncated_steps": 20}
    # task_id = "agent_multi_step_0"
    
    # AceBench-Multi-Turn (only FC support for now)
    # task_id agent_multi_turn_0-agent_multi_turn_29
    # env_name = "acebench_multi_turn"
    # env_config = {"domain": "agent_multi_turn", "user_model": "gpt-4.1", "user_provider": "openai", "truncated_steps": 20}
    # task_id = "agent_multi_turn_1"
    
    
    # Generate save file path based on environment type
    if env_name in ["bfcl", "envscaler_non_conversation_rl","envscaler_non_conversation_sft", "acebench_multi_step"]:
        save_file_path = f"result_debug/{env_name}-{agent_model}-{infer_mode}_{get_current_time()}.json"
    elif env_name in ["envscaler_conversation_rl","envscaler_conversation_sft", "acebench_multi_turn"]:
        save_file_path = f"result_debug/{env_name}-{agent_model}-{infer_mode}_{env_config['user_model']}_{get_current_time()}.json"
    else: # tau bench
        save_file_path = f"result_debug/{env_name}-{agent_model}-{infer_mode}_{env_config['user_model']}_{env_config['user_strategy']}_{get_current_time()}.json"
    
    # Run task solving
    result = solve_task(
        env_name = env_name,
        env_config = env_config,
        agent_model = agent_model,
        agent_model_provider = agent_model_provider,
        infer_mode = infer_mode,
        enable_thinking = enable_thinking,
        task_id = task_id
    )
    print("save_file_path:", save_file_path)
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    save_json(save_file_path, result)