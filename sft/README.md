# SFT Implementation

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 📋 Overview

We perform SFT training based on the [LlamaFactory framework](https://github.com/hiyouga/LlamaFactory). This folder contains data-processing scripts and training configurations:
- **Data-processing scripts**: Convert trajectory data into LlamaFactory training format
- **Training configuration**: Example Qwen3 model SFT training configuration

## 📁 Directory Structure

```
sft/
├── step1_process_messages_by_tool_template.py  # Step 1: Apply template
├── step2_process_llamafactory_format.py        # Step 2: rocess Messages
└── qwen3_full_sft.yaml                         # Training config file
```

## 🚀 Quick Start

### Step 1: Download the dataset

Download the EnvScaler SFT trajectory data from HuggingFace: [envscaler_sft_traj_9k_metadata](https://huggingface.co/datasets/XXHStudyHard/EnvScaler-SFT-Traj-9K)

### Step 2: Apply the Tool Template

- To ensure consistency with the model’s original tool-calling format during training, we first convert structured Messages (containing user, assistant, tool, content, reasoning_content, etc.) into Messages that only contain (user, assistant, content) according to the chat_template.  
- After this step, there is no need to use LlamaFactory’s tool-training mode or worry about tool-format alignment.

```bash
# Output: envscaler_sft_traj_9k_metadata_apply_qwen3_template.json
python sft/step1_process_messages_by_tool_template.py
```

### Step 3: Process Messages
- In Thinking mode, Qwen3 automatically removes the reasoning process of all turns when the chat_template is applied.
- To let the model learn the reasoning process of every single turn, we split an n-turn sample into n sub-samples (with turn counts 1, 2, ..., n).
- In each sub-sample, only the last turn’s output is supervised (achieved via LlamaFactory’s `mask_history` hyper-parameter).

```bash
# Output: alpaca_mask_history_envscaler_sft_traj_9k
python sft/step2_process_llamafactory_format.py
```

### Step 4: Install & configure LlamaFactory

Follow the [LlamaFactory official docs](https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html) to install the framework:

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git 
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

### Step 5: Configure dataset and training hyper-parameters

```bash
# Edit data/dataset_info.json
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
# Edit qwen3_full_sft.yaml config file, set necessary paths and parameters
```
Edit the `qwen3_full_sft.yaml` configuration file and set the required paths and parameters:

### Step 6: Launch training

Run training under the LlamaFactory project directory:

```bash
cd /path/to/LlamaFactory

# Copy the config file into the LlamaFactory project directory
cp /path/to/EnvScaler/sft/qwen3_full_sft.yaml .

# Start training with llamafactory-cli
llamafactory-cli train qwen3_full_sft.yaml
```

## 🔗 Related Resources

- [LlamaFactory project homepage](https://github.com/hiyouga/LlamaFactory)
- [LlamaFactory documentation](https://llamafactory.readthedocs.io/)
- [EnvScaler-SFT-Traj-9K dataset](https://huggingface.co/datasets/XXHStudyHard/EnvScaler-SFT-Traj-9K)