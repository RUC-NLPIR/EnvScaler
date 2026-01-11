# SkelBuilder

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 项目简介

`skel_builder` 是一个自动化环境骨架构建系统，用于从任务数据集中提取任务，并自动生成对应的状态化、领域特定的环境（Environment）骨架代码。该系统通过三个阶段的工作流程，将原始任务转换为完整的、可执行的 Python 环境类。

<p align="center">
    <img src="../Figs/skelbuilder_framework.png" width="98%"> <br>
  Framework of <b>SkelBuilder</b>.
</p>


## 项目结构

```
skel_builder/
├── stage1_collect_env_from_task/    # 阶段1: 从任务中收集环境
├── stage2_syn_env/                  # 阶段2: 合成环境代码
├── stage3_check_env/                # 阶段3: 检查并过滤环境
└── utils/                           # 工具函数
```

## 工作流程

### 阶段1: 从任务中收集环境 (stage1_collect_env_from_task)

从现有的指令遵循数据集中提取任务，并推断出对应的环境描述。

#### 步骤说明

1. **step0_collect_task.py** - 收集原始任务
   - 从 ToolAce 和 API-Bank 数据集中提取任务
   - 输出: `temp_result/step0_source_tasks.json`

2. **step1_judge_stateful_query.py** - 判断状态化查询
   - 判断任务是否依赖一个潜在的、状态化的、领域特定的环境
   - 输出: `temp_result/step1_stateful_task_judge.json`

3. **step2_infer_env_topic.py** - 推断环境主题
   - 给定一个任务，推断出最可能的状态化且领域特定的环境
   - 输出: `temp_result/step2_infered_env_description.json`

4. **step3_optional_get_embedding.py** (可选) - 获取嵌入向量
   - 为环境描述生成嵌入向量，用于后续的相似度计算和去重

5. **step3_optional_select_env.py** (可选) - 选择环境
   - 基于嵌入向量进行环境去重和筛选

#### 最终输出
- `final_result/env_description.json` - 环境描述数据

---

### 阶段2: 合成环境 (stage2_syn_env)

基于环境描述，生成完整的 Python 环境类代码。

#### 步骤说明

1. **step1_infer_state.py** - 推断状态空间
   - 给定环境描述和示例任务，推断出环境维护的状态变量（状态空间）
   - 定义实体（Entity）、属性（Attributes）和约束规则（Constraints & Rules）
   - 输出: `temp_result/step1_infer_state.json`

2. **step2_infer_state_code.py** - 生成状态代码
   - 将环境描述和状态空间定义转换为 Python 环境类定义
   - 生成包含 `__init__` 方法和状态属性的基础类结构
   - 使用 `TypedDict` 定义实体类型
   - 输出: `temp_result/step2_infer_state_code.json`

3. **step3_infer_operation.py** - 推断操作列表
   - 分析环境需要支持的操作列表
   - 将操作分为两类：
     - **信息查询类**（Information Query Class）
     - **状态修改类**（State Change Class）
   - 输出: `temp_result/step3_infer_operation.json`

4. **step4_infer_func_code.py** - 生成函数代码
   - 为每个操作生成对应的 Python 函数实现
   - 实现状态查询和状态修改的逻辑
   - 输出: `temp_result/step4_infer_func_code.json`

5. **step5_concat.py** - 拼接完整代码
   - 将环境类定义和所有方法拼接成完整的 Python 环境类代码
   - 进行 AST（抽象语法树）检查，确保代码语法正确
   - 输出: `temp_result/step5_env_class_code.json`

6. **step6_analysis_env_class_code.py** - 分析环境类代码
   - 分析生成的环境类代码
   - 提取工具模式（Tool Schema）和函数详情
   - 输出: `temp_result/step6_analysis_env_class_code.json`

#### 最终输出
- `final_result/env_with_code.json` - 包含完整环境类代码的数据

---

### 阶段3: 检查环境 (stage3_check_env)

对生成的环境进行测试和验证，过滤出高质量的环境。

#### 步骤说明

1. **step1_gen_test_config.py** - 生成测试配置
   - 为每个环境生成测试初始化状态（test initialization state）
   - 基于环境类定义生成符合约束的 JSON 配置
   - 输出: `temp_result/step1_gen_test_init_config.json`

2. **step2_roll_check.py** - 滚动检查
   - 实例化环境并运行功能测试
   - 检查环境的功能正确性和稳定性
   - 记录测试结果（通过、警告、失败）
   - 输出: `temp_result/step2_roll_check.json`

3. **step3_filter_env_by_check_result.py** - 根据检查结果过滤
   - 根据测试结果计算准确率指标：
     - `not_fail_acc`: 非失败准确率（通过 + 警告）/ 总数
     - `pass_acc`: 通过准确率（通过数）/ 总数
   - 根据阈值过滤环境
   - 精简环境元数据，生成最终的环境元数据
   - 输出: 
     - `temp_result/step3_selected_env_data.json`

#### 最终输出
- `final_result/filtered_env_metadata.json` - 过滤后的最终环境元数据
---

## 使用方法

### 运行完整流程

按照阶段顺序依次执行：

```bash
# 所有脚本使用相对路径，需要从项目根目录运行
cd skel_builder

# 阶段1: 收集环境
python stage1_collect_env_from_task/step0_collect_task.py
python stage1_collect_env_from_task/step1_judge_stateful_query.py
python stage1_collect_env_from_task/step2_infer_env_topic.py
# 可选: python stage1_collect_env_from_task/step3_optional_get_embedding.py
# 可选: python stage1_collect_env_from_task/step3_optional_select_env.py

# 阶段2: 合成环境
python stage2_syn_env/step1_infer_state.py
python stage2_syn_env/step2_infer_state_code.py
python stage2_syn_env/step3_infer_operation.py
python stage2_syn_env/step4_infer_func_code.py
python stage2_syn_env/step5_concat.py
python stage2_syn_env/step6_analysis_env_class_code.py

# 阶段3: 检查环境
python stage3_check_env/step1_gen_test_config.py
python stage3_check_env/step2_roll_check.py
python stage3_check_env/step3_filter_env_by_check_result.py
```

### 配置说明
- 确保已配置 OpenAI API 密钥 或者 实现自定义`llm_inference`函数在`utils/call_llm`文件下
- 各阶段的中间结果保存在 `temp_result/` 目录
- 最终结果保存在 `final_result/` 目录

---

## 数据格式

### 环境描述格式 (阶段1输出)

```json
{
  "task": "任务描述",
  "task_from": "api-bank",
  "environment_summary": "环境摘要",
  "environment_introduction": "环境介绍",
  "usefulness": 8,
  "modelability": 9
}
```

### 环境代码格式 (阶段2输出)

```json
{
  "environment_summary": "环境摘要",
  "environment_introduction": "环境介绍",
  "state_space_definition": [
    {
      "entity_name": "实体名称",
      "attributes": [
        {
          "name": "属性名",
          "type": "属性类型",
          "description": "属性描述"
        }
      ]
    }
  ],
  "constraints_rules": [
    "约束规则1",
    "约束规则2"
  ],
  "operation_list": [
    {
      "operation_name": "操作名称",
      "operation_type": "query|modify",
      "description": "操作描述",
      "code": "Python函数代码"
    }
  ],
  "env_class_name": "EnvironmentClassName",
  "env_class_code": "完整的Python环境类代码",
  "env_class_def": "环境类定义（从文件开始到__init__方法）",
  "env_structure": {
    "states": {
      "state_name": {
        "type": "状态类型",
        "used_by": ["使用该状态的方法列表"],
        "modified_by": ["修改该状态的方法列表"]
      }
    },
    "methods": {
      "method_name": {
        "calls": ["调用的方法列表"],
        "reads": ["读取的状态列表"],
        "writes": ["写入的状态列表"]
      }
    }
  },
  "env_func_details": {
    "function_name": {
      "source_code": "函数源代码",
      "doc": "函数文档字符串",
      "signature": {
        "parameters": [
          {
            "name": "参数名",
            "type": "参数类型"
          }
        ]
      }
    }
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "函数名",
        "description": "函数描述",
        "parameters": {
          "type": "object",
          "properties": {
            "param_name": {
              "type": "参数类型"
            }
          },
          "required": ["必需参数列表"]
        }
      }
    }
  ]
}
```

### 测试结果格式 (阶段3输出)

```json
{
  // ... 阶段2的所有字段 ...
  "init_config_list": [
    {
      // 测试初始化配置
    }
  ],
  "func_test_result": {
    "func_test_cases": {
      "summary": {
        "total_count": 50,           // 总测试用例数
        "pass_count": 35,            // 通过数
        "warning_count": 10,         // 警告数
        "fail_count": 5,             // 失败数
        "positive_count": 30,        // 正向测试用例数
        "negative_count": 20,        // 负向测试用例数
        "positive_pass_count": 25,   // 正向通过数
        "positive_warning_count": 3, // 正向警告数
        "positive_fail_count": 2,    // 正向失败数
        "negative_pass_count": 10,   // 负向通过数
        "negative_warning_count": 7, // 负向警告数
        "negative_fail_count": 3     // 负向失败数
      },
      "details": {
        "function_name": {
          "summary": {
            // 该函数的测试统计（与全局summary结构相同）
            "total_count": 5,
            "pass_count": 4,
            "warning_count": 1,
            "fail_count": 0,
            // ... 其他统计字段
          },
          "cases": [
            {
              "step": 0,                    // 测试步骤索引
              "case_type": "positive",      // 测试类型：positive/negative
              "status": "pass",             // 测试结果：pass/warning/fail
              "passed": true,               // 是否通过（兼容字段）
              "thought": "选择此方法的原因...", // Agent思考过程
              "parameters": {               // 函数调用参数
                "param1": "value1"
              },
              "state_before_call": {        // 调用前环境状态
                // 状态快照
              },
              "state_after_call": {         // 调用后环境状态
                // 状态快照
              },
              "state_diff": {               // 状态差异
                // 状态变化详情
              },
              "observation": {              // 函数返回值
                "success": true,
                "data": {}
              },
              "check_result": {             // 白盒检查结果
                "analysis": "方法行为分析...",
                "result": "pass",
                "error_reason": "No error"
              },
              "check_reason": "检查理由..." // 检查理由（从check_result提取）
            }
            // ... 更多测试用例
          ]
        }
        // ... 更多函数的测试详情
      }
    }
  }
}
```

---

## 许可证

本项目属于 EnvScaler 项目的一部分。

