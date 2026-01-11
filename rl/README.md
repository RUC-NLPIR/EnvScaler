# RL Implementation

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 📋 Overview
We perform RL training based on the [ROLL framework](https://github.com/alibaba/ROLL). This folder contains only the **additional** components that are grafted onto the ROLL repository:
- **Environment implementation**: the EnvScaler environment and the BFCL evaluation environment  
- **Environment manager**: a manager tailored for the ROLL framework  
- **Configuration files**: training configurations for EnvScaler  

## 📁 Directory Structure
(mirrors the ROLL layout)
```
rl/
├── example/                          # sample configuration files
│   └── env_scaler/
│       └── only_non_conv_qwen3_8gpu.yaml  # example training config
└── roll/                             # code to be integrated into ROLL
    └── pipeline/
        └── agentic/
            ├── env/                  # environments folder
            │   ├── envscaler_env/    # EnvScaler environment
            │   └── bfcl_env/         # BFCL evaluation environment
            └── env_manager/          # environment manager
                ├── traj_env_manager_for_env_scaler.py  # EnvScaler env manager
                └── traj_env_manager_for_env_scaler_util.py
```

## 🚀 Quick Start

### Step 1: Install the ROLL Framework

First, install the ROLL framework following the [official ROLL documentation](https://github.com/alibaba/ROLL):

```bash
# Clone the ROLL repo
git clone https://github.com/alibaba/ROLL.git 
cd ROLL

# Install and configure according to the ROLL docs
# See: https://alibaba.github.io/ROLL/docs/Getting%20Started/Installation/ 
```

### Step 2: Integrate the EnvScaler Environment Code

Add the new code into the ROLL project (it will **not** overwrite any existing ROLL code):
- The commands below only **add** new code; they do **not** replace or conflict with ROLL’s original environments or managers  
- You can also copy the files manually; the directory structure is one-to-one

```bash
# Assume EnvScaler is at /path/to/EnvScaler
# and ROLL is at /path/to/ROLL

# Add the new environment code (these are new environments, no conflicts)
cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env \
      /path/to/ROLL/roll/pipeline/agentic/env/

cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env/bfcl_env \
      /path/to/ROLL/roll/pipeline/agentic/env/

# Add the new environment manager
cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env_manager/traj_env_manager_for_env_scaler* \
      /path/to/ROLL/roll/pipeline/agentic/env_manager/
```

### Step 3: Register the New Environments

Append the following registration code to `roll/pipeline/agentic/env/__init__.py` (add at the end; it will not affect ROLL’s original registrations):

```python
import gem

# EnvScaler environments (newly added)
gem.register(env_id="envscaler_conv_env", 
             entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerConvRLEnv")
gem.register(env_id="envscaler_non_conv_env", 
             entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerNonConvRLEnv")

# BFCL environment (newly added)
gem.register(env_id="bfcl", 
             entry_point="roll.pipeline.agentic.env.bfcl_env:BfclEnv")
```

### Step 4: Prepare the Configuration File

Copy the config file into ROLL’s config directory:

```bash
# Copy the config into ROLL
mkdir -p /path/to/ROLL/examples/env_scaler
cp /path/to/EnvScaler/rl/example/env_scaler/only_non_conv_qwen3_8gpu.yaml \
   /path/to/ROLL/examples/env_scaler/
```
Edit the necessary paths and parameters (model paths, output directories, etc.) inside the config file.

### Step 5: Launch Training

Run training from the ROLL project root:

```bash
cd /path/to/ROLL

#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name only_non_conv_qwen3_8gpu
```

## 🔗 Related Resources

- [ROLL Project Homepage](https://github.com/alibaba/ROLL )
- [ROLL Documentation](https://alibaba.github.io/ROLL/docs/Overview/ )
- [ROLL Agentic Pipeline Documentation](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/agentic_pipeline_start)
- [ROLL Configuration Reference](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/config_system/)
- [Gem Documentation](https://github.com/axon-rl/gem/)