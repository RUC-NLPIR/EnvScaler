# Skel Builder

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## Project Overview

`skel_builder` is an **automated environment-skeleton construction system** that extracts tasks from a dataset and automatically generates the corresponding **stateful, domain-specific** Environment skeleton code. Through a **three-stage pipeline**, the system transforms raw tasks into complete, executable Python environment classes.

<p align="center">
    <img src="../Figs/skelbuilder_framework.png" width="98%"> <br>
  Framework of <b>SkelBuilder</b>.
</p>

## Project Structure

```
skel_builder/
├── stage1_collect_env_from_task/    # Stage 1: collect environments from tasks
├── stage2_syn_env/                  # Stage 2: synthesize environment code
├── stage3_check_env/                # Stage 3: check & filter environments
└── utils/                           # utility functions
```

## Workflow

### Stage 1 – Collect Environments from Tasks (`stage1_collect_env_from_task`)

Extract tasks from existing instruction-following datasets and infer the corresponding environment descriptions.

#### Step-by-step

1. **step0_collect_task.py** – collect raw tasks  
   - Extract tasks from ToolAce & API-Bank datasets  
   - Output: `temp_result/step0_source_tasks.json`

2. **step1_judge_stateful_query.py** – judge stateful queries  
   - Decide whether a task depends on a potential, stateful, domain-specific environment  
   - Output: `temp_result/step1_stateful_task_judge.json`

3. **step2_infer_env_topic.py** – infer environment topic  
   - Given a task, infer the most likely stateful & domain-specific environment  
   - Output: `temp_result/step2_infered_env_description.json`

4. **step3_optional_get_embedding.py** (optional) – obtain embeddings  
   - Generate embeddings for environment descriptions for later similarity / de-duplication

5. **step3_optional_select_env.py** (optional) – select environments  
   - De-duplicate & filter environments based on embeddings

#### Final output
- `final_result/env_description.json` – environment description data

---

### Stage 2 – Synthesize Environments (`stage2_syn_env`)

From environment descriptions, generate complete Python environment-class code.

#### Step-by-step

1. **step1_infer_state.py** – infer state space  
   - Given environment description & sample tasks, infer the state variables (state space) maintained by the environment  
   - Define Entities, Attributes, Constraints & Rules  
   - Output: `temp_result/step1_infer_state.json`

2. **step2_infer_state_code.py** – generate state code  
   - Convert environment description & state-space definition into a Python environment-class definition  
   - Produce a basic class structure with `__init__` and state attributes, using `TypedDict` for entity types  
   - Output: `temp_result/step2_infer_state_code.json`

3. **step3_infer_operation.py** – infer operation list  
   - Analyse the list of operations the environment must support, split into  
     - **Information Query Class**  
     - **State Change Class**  
   - Output: `temp_result/step3_infer_operation.json`

4. **step4_infer_func_code.py** – generate function code  
   - For each operation, generate the corresponding Python function implementation, covering state-query & state-modification logic  
   - Output: `temp_result/step4_infer_func_code.json`

5. **step5_concat.py** – concatenate full code  
   - Stitch the environment-class definition and all methods into a complete Python environment class  
   - Perform AST checks to guarantee syntactic correctness  
   - Output: `temp_result/step5_env_class_code.json`

6. **step6_analysis_env_class_code.py** – analyse environment-class code  
   - Analyse the generated environment-class code, extract tool schemas & function details  
   - Output: `temp_result/step6_analysis_env_class_code.json`

#### Final output
- `final_result/env_with_code.json` – data containing complete environment-class code

---

### Stage 3 – Check Environments (`stage3_check_env`)

Test & validate generated environments and filter out high-quality ones.

#### Step-by-step

1. **step1_gen_test_config.py** – generate test configurations  
   - For each environment, generate test initialization states (JSON configs conforming to constraints)  
   - Output: `temp_result/step1_gen_test_init_config.json`

2. **step2_roll_check.py** – rolling check  
   - Instantiate environments and run functional tests  
   - Check functional correctness & stability, record results (pass / warning / fail)  
   - Output: `temp_result/step2_roll_check.json`

3. **step3_filter_env_by_check_result.py** – filter by check results  
   - Compute accuracy metrics from test results  
     - `not_fail_acc` = (pass + warning) / total  
     - `pass_acc` = pass / total  
   - Filter environments by thresholds  
   - Trim environment metadata and produce final metadata  
   - Output :  `temp_result/step3_selected_env_data.json`

#### Final output
- `final_result/filtered_env_metadata.json` – final filtered environment metadata

---

## Usage

### Run the full pipeline

Execute stages sequentially from the project root:

```bash
# all scripts use relative paths – run from project root
cd skel_builder

# Stage 1 – collect environments
python stage1_collect_env_from_task/step0_collect_task.py
python stage1_collect_env_from_task/step1_judge_stateful_query.py
python stage1_collect_env_from_task/step2_infer_env_topic.py
# optional: python stage1_collect_env_from_task/step3_optional_get_embedding.py
# optional: python stage1_collect_env_from_task/step3_optional_select_env.py

# Stage 2 – synthesize environments
python stage2_syn_env/step1_infer_state.py
python stage2_syn_env/step2_infer_state_code.py
python stage2_syn_env/step3_infer_operation.py
python stage2_syn_env/step4_infer_func_code.py
python stage2_syn_env/step5_concat.py
python stage2_syn_env/step6_analysis_env_class_code.py

# Stage 3 – check environments
python stage3_check_env/step1_gen_test_config.py
python stage3_check_env/step2_roll_check.py
python stage3_check_env/step3_filter_env_by_check_result.py
```

### Configuration notes
- Ensure an OpenAI API key is configured OR implement a custom `llm_inference` function under `utils/call_llm`
- Intermediate results are saved in `temp_result/`
- Final results are saved in `final_result/`

---

## Data Formats

### Environment-description format (Stage 1 output)

```json
{
  "task": "task description",
  "task_from": "api-bank",
  "environment_summary": "environment summary",
  "environment_introduction": "environment introduction",
  "usefulness": 8,
  "modelability": 9
}
```

### Environment-code format (Stage 2 output)

```json
{
  "environment_summary": "environment summary",
  "environment_introduction": "environment introduction",
  "state_space_definition": [
    {
      "entity_name": "entity name",
      "attributes": [
        {
          "name": "attribute name",
          "type": "attribute type",
          "description": "attribute description"
        }
      ]
    }
  ],
  "constraints_rules": [
    "constraint rule 1",
    "constraint rule 2"
  ],
  "operation_list": [
    {
      "operation_name": "operation name",
      "operation_type": "query|modify",
      "description": "operation description",
      "code": "Python function code"
    }
  ],
  "env_class_name": "EnvironmentClassName",
  "env_class_code": "complete Python environment-class code",
  "env_class_def": "environment-class definition (from file start to __init__ method)",
  "env_structure": {
    "states": {
      "state_name": {
        "type": "state type",
        "used_by": ["list of methods using this state"],
        "modified_by": ["list of methods modifying this state"]
      }
    },
    "methods": {
      "method_name": {
        "calls": ["list of called methods"],
        "reads": ["list of read states"],
        "writes": ["list of written states"]
      }
    }
  },
  "env_func_details": {
    "function_name": {
      "source_code": "function source code",
      "doc": "function docstring",
      "signature": {
        "parameters": [
          {
            "name": "parameter name",
            "type": "parameter type"
          }
        ]
      }
    }
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "function name",
        "description": "function description",
        "parameters": {
          "type": "object",
          "properties": {
            "param_name": {
              "type": "parameter type"
            }
          },
          "required": ["list of required parameters"]
        }
      }
    }
  ]
}
```

### Test-result format (Stage 3 output)

```json
{
  // ... all Stage 2 fields ...
  "init_config_list": [
    {
      // test initialization config
    }
  ],
  "func_test_result": {
    "func_test_cases": {
      "summary": {
        "total_count": 50,           // total test cases
        "pass_count": 35,            // passed
        "warning_count": 10,         // warnings
        "fail_count": 5,             // failed
        "positive_count": 30,        // positive test cases
        "negative_count": 20,        // negative test cases
        "positive_pass_count": 25,   // positive passed
        "positive_warning_count": 3, // positive warnings
        "positive_fail_count": 2,    // positive failed
        "negative_pass_count": 10,   // negative passed
        "negative_warning_count": 7, // negative warnings
        "negative_fail_count": 3     // negative failed
      },
      "details": {
        "function_name": {
          "summary": {
            // same structure as global summary
            "total_count": 5,
            "pass_count": 4,
            "warning_count": 1,
            "fail_count": 0
            // ... other fields
          },
          "cases": [
            {
              "step": 0,                    // test-step index
              "case_type": "positive",      // positive / negative
              "status": "pass",             // pass / warning / fail
              "passed": true,               // compatibility flag
              "thought": "reason for choosing this method...", // agent thought
              "parameters": {               // function-call parameters
                "param1": "value1"
              },
              "state_before_call": {        // env state before call
                // state snapshot
              },
              "state_after_call": {         // env state after call
                // state snapshot
              },
              "state_diff": {               // state delta
                // state change details
              },
              "observation": {              // function return value
                "success": true,
                "data": {}
              },
              "check_result": {             // white-box check result
                "analysis": "analysis of method behaviour...",
                "result": "pass",
                "error_reason": "No error"
              },
              "check_reason": "reason for check..." // extracted from check_result
            }
            // ... more test cases
          ]
        }
        // ... more function test details
      }
    }
  }
}
```

---

## License

This project is part of the EnvScaler project.