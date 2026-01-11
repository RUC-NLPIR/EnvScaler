# SFT 实现

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 📋 概述

我们基于 [LlamaFactory框架](https://github.com/hiyouga/LlamaFactory) 进行SFT训练，该文件夹包含数据处理脚本和训练配置：
- **数据处理脚本**：将轨迹数据转换为LlamaFactory训练格式
- **训练配置**：Qwen3模型SFT训练配置示例

## 📁 目录结构

```
sft/
├── step1_process_messages_by_tool_template.py  # 步骤1：转换消息格式
├── step2_process_llamafactory_format.py        # 步骤2：转换为LlamaFactory格式
├── qwen3_full_sft.yaml                         # 训练配置文件
└── README_ZH.md                                # 本文件
```

## 🚀 快速开始

### 步骤1：下载数据

从HuggingFace下载EnvScaler SFT轨迹数据：[envscaler_sft_traj_9k_metadata](https://huggingface.co/datasets/XXHStudyHard/EnvScaler-SFT-Traj-9K)

### 步骤2：应用Tool Template

- 为了确保训练过程中与模型原始工具调用格式保持一致，我们首先将结构化Messages（包含user、assistant、tool、content、reasoning_content等）根据 chat_template 转换为仅包含（user, assistant, content）的Messages。
- 处理后，无需使用 LlamaFactory 的工具训练模式和担心工具格式对齐问题。

```bash
# 输出: envscaler_sft_traj_9k_metadata_apply_qwen3_template.json
python sft/step1_process_messages_by_tool_template.py
```

### 步骤3: 处理Messages
- 在Thinking模式下，Qwen3 在应用 chat_template 时会自动移除所有轮次的推理过程。
- 为了让模型学习每一回合的推理过程，我们将一个 n 回合的样本拆分为 n 个子样本（对应的回合数为 1, 2, ..., n）。
- 每个子样本中仅对最后一轮的输出进行监督（通过 LlamaFactory 中的 `mask_history` 超参数实现）。

```bash
# 输出: alpaca_mask_history_envscaler_sft_traj_9k
python sft/step2_process_llamafactory_format.py
```

### 步骤4：安装配置LlamaFactory

按照[LlamaFactory官方文档](https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html)安装框架：

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

### 步骤5：配置数据集和训练参数

```bash
# 编辑 data/dataset_info.json
"alpaca_mask_history_envscaler_sft_traj_9k": {
  "file_name": "your_path/alpaca_mask_history_envscaler_sft_traj_9k.json",
  "formatting": "alpaca",
  "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "history": "history",
      "system": "system"
    }
  }
# 编辑qwen3_full_sft.yaml配置文件, 设置必要的路径和参数
```
编辑 `qwen3_full_sft.yaml` 配置文件，设置必要的路径和参数：

### 步骤6：启动训练

在LlamaFactory项目目录下运行训练：

```bash
cd /path/to/LlamaFactory

# 将配置文件复制到LlamaFactory项目目录
cp /path/to/EnvScaler/sft/qwen3_full_sft.yaml .

# 使用llamafactory-cli启动训练
llamafactory-cli train qwen3_full_sft.yaml
```

## 🔗 相关资源

- [LlamaFactory项目主页](https://github.com/hiyouga/LlamaFactory)
- [LlamaFactory文档](https://llamafactory.readthedocs.io/)
- [EnvScaler-SFT-Traj-9K数据集](https://huggingface.co/datasets/XXHStudyHard/EnvScaler-SFT-Traj-9K)

