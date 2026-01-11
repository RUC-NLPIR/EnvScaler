# Interact with Environment

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

This project implements Agent–Environment interaction and is mainly used for:
1. **Collecting training trajectories** by interacting with the EnvScaler synthetic environment  
2. **Evaluating performance** on benchmark environments such as TauBench and AceBench
3. **Used as a validation and monitoring environment in RL training**

<p align="center">
    <img src="../Figs/interact_with_env.png" width="30%"> <br>
  Diagram of <b>Agent-Env Interaction</b>.
</p>


## 📁 Directory Structure

```
interact_with_env/
├── run_main.py              # Main script: batch task execution
├── run_main_debug.py        # Debug script: single-task execution & debugging
├── traj_filter.py           # SFT trajectory filtering script
├── calc_avg_score.py        # Compute average reward score
├── agent/                   # Agent module
│   ├── task_solve_agent.py      # LLM Agent logic implementation
│   ├── agent_llm_inference.py   # LLM inference interface
│   └── system_prompt_util.py    # Agent system prompts
├── envscaler_env/           # EnvScaler environments (for training)
├── taubench_env/            # TauBench environments (for evaluation)
├── acebench_env/            # AceBench-Agent environments (for evaluation)
├── bfcl_env/                # BFCL Multi-Turn-Base environment (training monitor)
├── result/                  # Output directory for results
└── result_debug/            # Output directory for debug results
```

## 🔄 Agent–Environment Interaction Flow

Agent–Environment interaction follows the style of [Gym](https://github.com/openai/gym) / [Gem](https://github.com/axon-rl/gem), so you can easily migrate environments to any Gym/Gem-style training framework.  
The following is an abstract interaction flow:

```python
# Reset the environment and obtain the observation (task info or user utterance)
observation, info = Env.reset(task_index=0)

max_step = 30
cur_step = 0
# Interaction loop
while True:
    cur_step += 1
    # action: tool call or response to the user
    action = agent.step(observation)
    # observation: tool execution result or user reply
    observation, reward, terminated, truncated, info = Env.step(action)

    # Check termination
    if terminated or truncated or cur_step > max_step:
        break
```

After a run finishes, results are saved as a JSON file containing:

```json
{
    "task_info": {
        // Task-related information (varies by environment)
    },
    "tools": [
        // List of available tools
    ],
    "messages": [
        // Full conversation history
    ],
    "user_messages": [
        // User messages only (conversation environments)
    ],
    "trajectory": [
        // Detailed step-by-step log
        {
            "step": 0,
            "observation": {...},
            "action": {...},
            "reward": 0.0,
            "terminated": false,
            "truncated": false
        }
    ],
    "total_reward": 0.0,
    "terminated": false,
    "truncated": false,
    "final_observation": {...},
    "final_info": {...},
    "steps": 10
}
```

## 🔧 Supported Environments

### 1. EnvScaler Training Environments
```python
# EnvScaler non-conversation RL environment
env_name = "envscaler_non_conversation_rl"
env_config = {...}

# EnvScaler conversation RL environment
env_name = "envscaler_conversation_rl"
env_config = {...}

# EnvScaler non-conversation SFT environment
env_name = "envscaler_non_conversation_sft"
env_config = {...}

# EnvScaler conversation SFT environment
env_name = "envscaler_conversation_sft"
env_config = {...}
```
- We use `envscaler_non_conversation_sft` and `envscaler_conversation_sft` to collect SFT training trajectories.  
- We migrate `envscaler_non_conversation_rl` and `envscaler_conversation_rl` to the ROLL framework for RL training.

### 2. TauBench Evaluation Environments
```python
# TauBench retail domain
env_name = "tau_bench_retail"
env_config = {...}

# TauBench airline domain
env_name = "tau_bench_airline"
env_config = {...}
```
- We use `tau_bench_retail` and `tau_bench_airline` for the experiments reported in our paper.  
- We reorganized the official [TauBench](https://github.com/sierra-research/tau-bench) code while preserving the original logic as much as possible.

### 3. AceBench-Agent Evaluation Environments
```python
# AceBench multi-step environment
env_name = "acebench_multi_step"
env_config = {...}

# AceBench multi-turn environment
env_name = "acebench_multi_turn"
env_config = {...}
```
- We use `acebench_multi_step` and `acebench_multi_turn` for the experiments reported in our paper.  
- We reorganized the official [AceBench](https://github.com/chenchen0103/ACEBench) code while preserving the original logic as much as possible.  
- Note: ACEBench originally uses the `[func_name(param)]` prompt format; we modified the official code to support the LLM’s native function-calling (FC) interface for consistency.

### 4. BFCL Validation Environment
```python
env_name = "bfcl"
env_config = {"mode": "multi_turn_base", ...}
```
- We use the `bfcl (multi_turn_base)` environment for validation/monitoring during RL training.  
- For the experiments in our paper we used the official [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) code (`base`, `miss-param`, `miss-func`, `long-context`).

## 🚀 Quick Start

```bash
# All scripts use relative paths; run from the interact_with_env directory
cd interact_with_env
```

**Run the main script:**

```bash
# Batch processing of multiple tasks with multi-threading
python run_main.py
```
You need to edit the following settings in `run_main.py`:

```python
# 1. Agent model config
agent_model = "gpt-4.1"           # Model name to use
agent_model_provider = "openai"   # Model provider
enable_thinking = True            # Enable thinking mode
num_workers = 3                   # Number of parallel workers

# 2. Environment config
env_name = "selected_env_name"
infer_mode = "fc"                 # "prompt" or "fc"
env_config = {...}
task_ids = [i for i in range(...)]  # List of task IDs
```

**Debug mode:**
```bash
# Single-task debugging:
python run_main_debug.py
```

---

## License

This project is part of the EnvScaler project.