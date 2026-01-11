# RL Implementation

<div align="left">
  <a href="README_ZH.md">中文</a> | <a href="README.md">English</a>
</div>

## 📋 概述
我们基于 [ROLL框架](https://github.com/alibaba/ROLL) 进行RL训练, 该文件夹仅包含在ROLL仓库上新增的内容：
- **环境实现**：EnvScaler环境和BFCL评估环境
- **环境管理器**：用于ROLL框架的环境管理器
- **配置文件**：EnvScaler训练配置

## 📁 目录结构
(与ROLL结构对应)
```
rl/
├── example/                          # 配置文件示例
│   └── env_scaler/
│       └── only_non_conv_qwen3_8gpu.yaml  # 训练配置示例
└── roll/                             # 要集成到ROLL项目的代码
    └── pipeline/
        └── agentic/
            ├── env/                  # 环境目录
            │   ├── envscaler_env/    # EnvScaler环境
            │   └── bfcl_env/         # BFCL评估环境
            └── env_manager/          # 环境管理器
                ├── traj_env_manager_for_env_scaler.py  # EnvScaler环境管理器
                └── traj_env_manager_for_env_scaler_util.py
```


## 🚀 快速开始

### 步骤1：安装ROLL框架

首先，按照[ROLL官方文档](https://github.com/alibaba/ROLL)安装ROLL框架：

```bash
# 克隆ROLL仓库
git clone https://github.com/alibaba/ROLL.git
cd ROLL

# 按照ROLL文档进行安装和配置
# 参考：https://alibaba.github.io/ROLL/docs/Getting%20Started/Installation/
```

### 步骤2：集成EnvScaler环境代码

将本项目的代码新增到ROLL项目中（不会替换ROLL框架原有代码）：
- 以上操作是在ROLL框架中新增代码，不会替换或覆盖ROLL原有的环境和环境管理器
- 您也可以手动复制迁移, 路径是一一对应的

```bash
# 假设您的EnvScaler项目路径为 /path/to/EnvScaler
# ROLL项目路径为 /path/to/ROLL

# 新增环境代码到ROLL项目（这些是新环境，不会与ROLL原有环境冲突）
cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env/envscaler_env \
      /path/to/ROLL/roll/pipeline/agentic/env/

cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env/bfcl_env \
      /path/to/ROLL/roll/pipeline/agentic/env/

# 新增环境管理器到ROLL项目
cp -r /path/to/EnvScaler/rl/roll/pipeline/agentic/env_manager/traj_env_manager_for_env_scaler* \
      /path/to/ROLL/roll/pipeline/agentic/env_manager/
```

### 步骤3：添加环境注册

在 `roll/pipeline/agentic/env/__init__.py` 文件中新增环境注册代码（在文件末尾添加即可，不会影响ROLL原有的环境注册）：

```python
import gem

# EnvScaler环境注册（新增）
gem.register(env_id="envscaler_conv_env", 
             entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerConvRLEnv")
gem.register(env_id="envscaler_non_conv_env", 
             entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerNonConvRLEnv")

# BFCL环境注册（新增）
gem.register(env_id="bfcl", 
             entry_point="roll.pipeline.agentic.env.bfcl_env:BfclEnv")
```

### 步骤4：准备配置文件

将配置文件复制到ROLL项目的配置目录：

```bash
# 复制配置文件到ROLL项目
mkdir -p /path/to/ROLL/examples/env_scaler
cp /path/to/EnvScaler/rl/example/env_scaler/only_non_conv_qwen3_8gpu.yaml \
   /path/to/ROLL/examples/env_scaler/
```
设置必要的路径和参数：编辑配置文件中的模型和输出目录。

### 步骤5：启动训练

在ROLL项目目录下运行训练：

```bash
cd /path/to/ROLL

#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name only_non_conv_qwen3_8gpu
```

## 🔗 相关资源

- [ROLL项目主页](https://github.com/alibaba/ROLL)
- [ROLL文档](https://alibaba.github.io/ROLL/docs/Overview/)
- [ROLL Agentic Pipeline文档](https://alibaba.github.io/ROLL/docs/User%20Guides/Pipeline/agentic_pipeline_start)
- [ROLL 参数介绍](https://alibaba.github.io/ROLL/docs/User%20Guides/Configuration/config_system/)
- [Gem介绍](https://github.com/axon-rl/gem/)
