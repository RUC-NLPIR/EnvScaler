# 评估

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

我们在BFCL-v3 Multi-Turn, Tau-Bench, ACEBench-Agent基准进行评估实验 (在推理模式下，使用LLM自身的FC接口)。

## 评估基准

### 1. BFCL Multi-Turn

#### 评估说明

- **评估代码**: 使用[官方评估代码](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **推理模式**: FC 模式
- **评估配置**: Thinking 模式，温度系数 0.7
- **数据版本**: 
  - 由于数据版本变更，BFCL-v4 的 multi-turn 数据与 BFCL-v3 的 multi-turn 略有差异，可能带来评估上的差异
  - 我们提供 v3 数据在 `bfcl_eval/data/` 目录（更新时间: 2024-09-22），您可以使用该文件夹替换官方数据
- **上下文长度**: 为适应 BFCL-v3 的 Long-Context 子集，我们设置最大上下文长度为 64K

#### 数据集信息

详细的 BFCL 数据集说明请参考 [`bfcl_eval/data/README.md`](bfcl_eval/data/README.md)。

---

### 2. Tau-Bench

> 💡 **提示**: 有关环境交互的详细说明，请参阅 [`interact_with_env/README.md`](../interact_with_env/README.md)

#### 运行评估

```bash
cd interact_with_env
python run_main.py
```

#### 环境配置

**Tau-Bench Retail**:
```python
env_name = "tau_bench_retail"
infer_mode = "fc"
env_config = {
    "mode": "eval", 
    "user_model": "gpt-4.1-2025-04-14", 
    "user_strategy": "llm_react",
    "user_provider": "openai",
}
task_ids = [i for i in range(115)]
```

**Tau-Bench Airline**:
```python
env_name = "tau_bench_airline"
infer_mode = "fc"
env_config = {
    "mode": "eval", 
    "user_model": "gpt-4.1-2025-04-14", 
    "user_strategy": "llm_react",
    "user_provider": "openai",
}
task_ids = [i for i in range(50)]
```

#### 评估说明

- **推理模式**: FC 模式，使用模型自身的函数调用接口
- **评估配置**: Thinking 模式，温度系数 0.7
- **用户模拟**: 使用 `gpt-4.1-2025-04-14` 模拟用户，策略为 `llm_react`
- **注意事项**: 由于测试任务较少以及引入 LLM 模拟用户，TauBench 的评估结果波动较大, **建议多次运行实验并取平均/中值**

---

### 3. ACEBench Agent


#### 运行评估

```bash
cd interact_with_env
python run_main.py
```

#### 环境配置

**ACEBench Multi-Step**:
```python
env_name = "acebench_multi_step"
infer_mode = "fc"
env_config = {
    "domain": "agent_multi_step", 
    "truncated_steps": 20
}
task_ids = [f"agent_multi_step_{i}" for i in range(20)]
```

**ACEBench Multi-Turn**:
```python
env_name = "acebench_multi_turn"
infer_mode = "fc"
env_config = {
    "domain": "agent_multi_turn", 
    "user_model": "gpt-4.1-2025-04-14", 
    "user_provider": "openai", 
    "truncated_steps": 20
}
task_ids = [f"agent_multi_turn_{i}" for i in range(30)]
```

#### 评估说明

- **格式适配**: ACEBench 原本使用 `[func_name(param)]` prompt 格式，我们修改了官方代码以支持 LLM 的原生函数调用接口 (FC)，确保一致性
- **评估配置**: Thinking 模式，温度系数 0.7
- **用户模拟**: Multi-Turn 子集使用 `gpt-4.1-2025-04-14` 模拟用户
- **注意事项**: 由于测试任务较少以及引入 LLM 模拟用户，ACEBench 的评估结果波动较大, **建议多次运行实验并取平均/中值**

---

## 结果分析

### 查看评估结果

评估结果保存在 `interact_with_env/result/` 目录下，每个任务的结果为一个 JSON 文件。结果文件包含以下关键字段：

```json
[
    {
        "total_reward": 0.0,        // 总奖励分数
        "terminated": false,        // 是否正常终止
        "truncated": false,         // 是否被截断
        "steps": 10,                // 交互步数
        "trajectory": [...]         // 完整的交互轨迹
    }
...
]
```

### 计算平均分数

```bash
cd interact_with_env
python calc_avg_score.py
```


---

## 相关链接

- [环境交互指南](../interact_with_env/README_ZH.md) - 详细了解如何与环境交互
- [BFCL 官方仓库](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) - BFCL 官方评估代码
- [TauBench 官方仓库](https://github.com/sierra-research/tau-bench) - TauBench 原始实现
- [ACEBench 官方仓库](https://github.com/chenchen0103/ACEBench) - ACEBench 原始实现