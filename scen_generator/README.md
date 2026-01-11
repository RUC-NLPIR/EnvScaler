# ScenGenerator

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## Project Overview

`scen_generator` is an automated scenario-generation workflow that produces multiple task-oriented scenarios (initial data-state configurations, tasks, and task-checking functions) for existing environment skeletons.

<p align="center">
    <img src="../Figs/scengenerator_framework.png" width="98%"> <br>
  Framework of <b>ScenGenerator</b>.
</p>

## Project Structure

```
scen_generator/
├── step1_gen_env_config.py         # Step 1: Generate initial environment configs
├── step2_gen_scenario_task.py      # Step 2: Generate scenario tasks
├── step3_gen_task_check_func.py    # Step 3: Generate task-checking functions
├── task_check_util/                # Task-checking utilities
│   ├── gen_checklist.py            # Generate checklists
│   └── gen_check_func.py           # Generate checking functions
├── utils/                          # Utility functions
│   ├── call_llm.py                 # LLM API wrapper
│   ├── process_file.py             # File I/O helpers
│   ├── auto_env.py                 # Environment-building helpers
│   └── util.py                     # Other helpers
├── temp_result/                    # Intermediate results
└── final_result/                   # Final results
```

## Workflow

Scenario generation proceeds through three sequential steps:

### Step 1: Generate Initial Environment Configs (`step1_gen_env_config.py`)

Create multiple initial-state configurations for each environment so that later steps can instantiate concrete scenarios.

- Input: Environment metadata file `filtered_env_metadata.json`
- Output: Per-environment list of initial configs → `temp_result/step1_init_env_config.json`

---

### Step 2: Generate Scenario Tasks (`step2_gen_scenario_task.py`)

Produce a natural-language task description for every initial config generated in Step 1.

- Input: Environment metadata + Step 1 file `temp_result/step1_init_env_config.json`
- Output: Scenario data with tasks → `temp_result/step2_gen_task.json`

---

### Step 3: Generate Task-Checking Functions (`step3_gen_task_check_func.py`)

For every task, emit a checklist and the corresponding Python function that decides whether the task has been solved.

- Input: Environment metadata + Step 2 file `temp_result/step2_gen_task.json`
- Output: Full scenario data with checklists & functions → `temp_result/step3_gen_task_check_func.json`, and the final consolidated file `final_result/env_scenario.json`

---

## Usage

### Run the Full Pipeline

Execute the scripts in order from the `scen_generator` directory:

```bash
# All scripts use relative paths—run from scen_generator/
cd scen_generator

# Step 1: Generate initial environment configs
python step1_gen_env_config.py

# Step 2: Generate scenario tasks
python step2_gen_scenario_task.py

# Step 3: Generate task-checking functions
python step3_gen_task_check_func.py
```

### Configuration Notes
- Provide an OpenAI API key (or implement your own `llm_inference` in `utils/call_llm.py`)
- Intermediate results live in `temp_result/`
- Final results are written to `final_result/`

---

## Data Formats

### Step 1 Output Format

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "init_config_list": [
      {...initial config 1...},
      {...initial config 2...},
      ...
    ]
  }
]
```

### Step 2 Output Format

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "task_id": "timestamp_001",
    "init_config": {...initial config...},
    "task": "Task description text..."
  }
]
```

### Step 3 Output Format (final scenario data)

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "task_id": "timestamp_001",
    "init_config": {...initial config...},
    "task": "Task description text...",
    "checklist": [
      "Has the new device DEV-9Z88H been registered?",
      ...
    ],
    "checklist_with_func": [
      {
        "check_item": "Has the new device DEV-9Z88H been registered?",
        "check_func": "def check_func(final_state):\n    ..."
      },
      ...
    ]
  }
]
```

---

## License

This project is part of EnvScaler.