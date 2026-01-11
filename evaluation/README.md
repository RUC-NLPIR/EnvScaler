# Evaluation

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

We conduct evaluation experiments on the BFCL-v3 Multi-Turn, Tau-Bench, and ACEBench-Agent benchmarks (in reasoning mode, using the LLM’s native FC interface).

## Evaluation Benchmarks

### 1. BFCL Multi-Turn

#### Evaluation Notes

- **Evaluation code**: Use the [official evaluation code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- **Inference mode**: FC mode
- **Evaluation config**: Thinking mode, temperature 0.7
- **Data version**:
  - Due to data-version changes, BFCL-v4 multi-turn data differs slightly from BFCL-v3 multi-turn, which may cause evaluation discrepancies
  - We provide v3 data in the `bfcl_eval/data/` directory (last updated 2024-09-22); you can replace the official data with this folder
- **Context length**: To accommodate BFCL-v3’s Long-Context subset, we set the maximum context length to 64 K

#### Dataset Information

See [`bfcl_eval/data/README.md`](bfcl_eval/data/README.md) for detailed BFCL dataset descriptions.

---

### 2. Tau-Bench

> 💡 **Tip**: For detailed instructions on environment interaction, see [`interact_with_env/README.md`](../interact_with_env/README.md)

#### Run Evaluation

```bash
cd interact_with_env
python run_main.py
```

#### Environment Configuration

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

#### Evaluation Notes

- **Inference mode**: FC mode, using the model’s native function-calling interface
- **Evaluation config**: Thinking mode, temperature 0.7
- **User simulation**: Uses `gpt-4.1-2025-04-14` to simulate the user with strategy `llm_react`
- **Caveat**: Because the test set is small and an LLM simulates the user, Tau-Bench results fluctuate; **we recommend running the experiment multiple times and reporting the average / median**

---

### 3. ACEBench Agent

#### Run Evaluation

```bash
cd interact_with_env
python run_main.py
```

#### Environment Configuration

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

#### Evaluation Notes

- **Format adaptation**: ACEBench originally uses the `[func_name(param)]` prompt format; we patched the official code to support the LLM’s native function-calling interface (FC) for consistency
- **Evaluation config**: Thinking mode, temperature 0.7
- **User simulation**: Multi-Turn subset uses `gpt-4.1-2025-04-14` to simulate the user
- **Caveat**: Because the test set is small and an LLM simulates the user, ACEBench results fluctuate; **we recommend running the experiment multiple times and reporting the average / median**

---

## Results Analysis

### View Evaluation Results

Evaluation results are saved in the `interact_with_env/result/` directory; each task produces one JSON file. Key fields in the result file:

```json
[
    {
        "total_reward": 0.0,        // total reward score
        "terminated": false,        // whether terminated normally
        "truncated": false,         // whether truncated
        "steps": 10,                // number of interaction steps
        "trajectory": [...]         // complete interaction trajectory
    }
...
]
```

### Compute Average Score

```bash
cd interact_with_env
python calc_avg_score.py
```

---

## Related Links

- [Environment Interaction Guide](../interact_with_env/README_ZH.md) – detailed instructions on interacting with environments
- [BFCL Official Repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) – BFCL official evaluation code
- [TauBench Official Repo](https://github.com/sierra-research/tau-bench) – original TauBench implementation
- [ACEBench Official Repo](https://github.com/chenchen0103/ACEBench) – original ACEBench implementation