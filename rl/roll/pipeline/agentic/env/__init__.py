"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""
import gem

from roll.utils.logging import get_logger
logger = get_logger()

# add envscaler env
gem.register(env_id="envscaler_conv_env", entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerConvRLEnv")
gem.register(env_id="envscaler_non_conv_env", entry_point="roll.pipeline.agentic.env.envscaler_env:EnvScalerNonConvRLEnv")

# add bfcl env
gem.register(env_id="bfcl", entry_point="roll.pipeline.agentic.env.bfcl_env:BfclEnv")