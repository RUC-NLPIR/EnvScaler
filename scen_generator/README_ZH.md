# ScenGenerator

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 项目简介

`scen_generator` 是一个自动化场景生成框架，用于为已有的环境骨架生成若干个任务场景(初始数据状态配置、任务和任务检查函数)。

<p align="center">
    <img src="../Figs/scengenerator_framework.png" width="98%"> <br>
  Framework of <b>ScenGenerator</b>.
</p>

## 项目结构

```
scen_generator/
├── step1_gen_env_config.py         # 步骤1: 生成环境初始配置
├── step2_gen_scenario_task.py      # 步骤2: 生成场景任务
├── step3_gen_task_check_func.py    # 步骤3: 生成任务检查函数
├── task_check_util/                # 任务检查工具模块
│   ├── gen_checklist.py            # 生成检查清单
│   └── gen_check_func.py           # 生成检查函数
├── utils/                          # 工具函数
│   ├── call_llm.py                 # LLM API 调用封装
│   ├── process_file.py             # 文件读写工具
│   ├── auto_env.py                 # 环境构建工具
│   └── util.py                     # 其他工具函数
├── temp_result/                    # 临时结果目录
└── final_result/                   # 最终结果目录
```

## 工作流程

场景生成过程分为三个主要步骤，需要按顺序执行：

### 步骤1: 生成环境初始配置 (`step1_gen_env_config.py`)

为每个环境生成多个初始状态配置，用于后续场景实例化。

- 输入: 环境元数据文件 `filtered_env_metadata.json`
- 输出: 每个环境的初始配置列表`temp_result/step1_init_env_config.json`

---

### 步骤2: 生成场景任务 (`step2_gen_scenario_task.py`)

基于环境初始配置，为每个场景生成对应的任务描述。
- 输入: 环境元数据文件, 以及步骤1生成的初始配置文件`temp_result/step1_init_env_config.json`
- 输出: 包含任务描述的场景数据`temp_result/step2_gen_task.json`

---

### 步骤3: 生成任务检查函数 (`step3_gen_task_check_func.py`)

为每个任务生成检查清单和对应的 Python 检查函数，用于验证任务是否成功完成。

- 输入: 环境元数据文件, 任务数据文件`temp_result/step2_gen_task.json`
- 输出: 包含检查清单和检查函数的完整场景数据`temp_result/step3_gen_task_check_func.json`, 最终数据`final_result/env_scenario.json`

---

## 使用方法

### 运行完整流程

按照步骤顺序依次执行：

```bash
# 所有脚本使用相对路径，需要从 `scen_generator` 目录运行
cd scen_generator

# 步骤1: 生成环境初始配置
python step1_gen_env_config.py

# 步骤2: 生成场景任务
python step2_gen_scenario_task.py

# 步骤3: 生成任务检查函数
python step3_gen_task_check_func.py
```

### 配置说明
- 确保已配置 OpenAI API 密钥 或者 实现自定义`llm_inference`函数在`utils/call_llm`文件下
- 中间结果保存在 `temp_result/` 目录
- 最终结果保存在 `final_result/` 目录

---

## 数据格式

### 步骤1输出格式

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "init_config_list": [
      {...初始配置1...},
      {...初始配置2...},
      ...
    ]
  }
]
```

### 步骤2输出格式

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "task_id": "timestamp_001",
    "init_config": {...初始配置...},
    "task": "任务描述文本..."
  }
]
```

### 步骤3输出格式（最终场景数据）

```json
[
  {
    "env_id": "env_001",
    "env_class_name": "HospitalEnv",
    "task_id": "timestamp_001",
    "init_config": {...初始配置...},
    "task": "任务描述文本...",
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

## 许可证

本项目属于 EnvScaler 项目的一部分。
