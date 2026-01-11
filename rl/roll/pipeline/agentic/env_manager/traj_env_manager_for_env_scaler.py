"""
roll/pipeline/agentic/env_manager/traj_env_manager_for_env_scaler.py
Trajectory environment manager for env_scaler.
Ensure enable_thinking=True for custom_apply_chat_template.
"""
import copy
from contextlib import nullcontext
from threading import Lock
from typing import Optional
import gem
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache, BaseEnvManager
from roll.utils.env_action_limiter import get_global_limiter
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.pipeline.agentic.tools.tool_env_wrapper import tool_wrapper
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field

from roll.pipeline.agentic.env_manager.traj_env_manager_for_env_scaler_util import non_conversational_system_prompt, conversational_system_prompt, merge_tools_into_system_prompt



def construct_system_prompt(env_name, env_introduction, tools) -> str:
    """Construct system prompt based on environment type (conversational or non-conversational)."""
    if env_name in ["envscaler_conversation_rl"] :
        system_prompt = conversational_system_prompt.format(env_introduction=env_introduction)
    elif env_name in ["envscaler_non_conversation_rl", "bfcl"]:
        system_prompt = non_conversational_system_prompt.format(env_introduction=env_introduction)   
    else:
        raise RuntimeError(f"Unknown env_name: {env_name}")  
    
    # If env introduction is provided, add it to system prompt  
    if env_introduction:
        system_prompt = f"{system_prompt}\n\nThe following is an introduction to the current environment:\n{env_introduction}"
        
    # In Prompt mode, merge tool information into system prompt
    system_prompt = merge_tools_into_system_prompt(system_prompt=system_prompt, tools=tools)
    return system_prompt



class TrajEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        """
        Initialize trajectory environment manager.
        """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RequestScheduler = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.running = False
        self.use_thread_lock = self.env_config.get("use_thread_lock", False)
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        with self.thread_lock, self.env_step_limiter:
            if "seed" in self.env_config['config']:
                self.env_config['config']["seed"] = self.env_config['group_seed']
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])
            if "tool_wrapper" in self.env_config:
                self.env = tool_wrapper(self.env,
                                        wrapper_args=self.env_config.tool_wrapper.wrapper_args,
                                        tool_configs=self.env_config.tool_wrapper.tool_configs)


        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def run_rollout_loop(self, data: DataProto):
        """
        Main rollout loop for trajectory collection.
        Runs multiple episodes until receiving stop signal.
        """
        # Initialize loop state
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        
        # Reset environment, start first episode
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        # Initialize performance statistics dictionary (used to record performance metrics for each episode)
        log_stats = {"generate_time": [], "step_time": [], "current_step": []}

        # Main loop - continue running episodes until stop signal is received
        # Loop condition: running=True and rollout_cache is not None (indicates there are episodes to run)
        while self.running and rollout_cache is not None:
            # Decision stage (LLM generates action)
            with Timer(name="generate", logger=None) as generate_timer:
                # Call make_decision: format input -> call LLM to generate response
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            # Environment interaction stage (execute action)
            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)

            log_stats["step_time"].append(step_timer.last)

            # Check episode termination conditions
            # If episode ends (environment terminated or reaches maximum length), process complete trajectory
            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                # Reset statistics dictionary, prepare to record statistics for next episode
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}

                # Format complete episode trajectory for training data
                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                
                # Generate trajectory identifier
                # traj_group_id: used to identify trajectories within the same group
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                # traj_id: used to uniquely identify each trajectory
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                
                # Add trajectory ID to rollout data
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                
                # Send formatted trajectory data to output queue
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout))

                # Reset environment, start next episode
                rollout_cache = self.reset()
                start_step = self.current_step

        # Loop ends, send end signal
        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None))

    def reset(self) -> RolloutCache:
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(self.env_config['group_id']))
        if self.episode_id is None:
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id

        with self.thread_lock, self.env_step_limiter:
            # `observation` describes the current game-state prompt;
            # `info["suffix"]` carries the current environment-specific state string.
            observation, info = self.env.reset(seed=seed)
            # Build system prompt
            self.agent_system_template = construct_system_prompt(
                env_name=info["env_name"],
                env_introduction=info["env_introduction"],
                tools=info["tools"]
            )
            if observation is None:
                return None
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,     # agent input messages
            **info,
        })
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        """
        Execute environment step with LLM-generated action.
        Decodes response, applies action, updates cache, and handles termination.
        """
        # Decode LLM-generated response
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)

        # Execute environment step (thread-safe)
        with self.thread_lock, self.env_step_limiter:
            observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        
        suffix = info.pop("suffix", None)

        # Update trajectory step and termination state
        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        
        # Check if maximum step limit is reached
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            # If environment itself is not marked as terminated, mark as truncated (because step limit is reached)
            if not terminated:
                self.rollout_cache.truncated = True
        
        # Update previous step's reward and response information
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = responses[0]
        
        # If environment returns additional information, update to history
        if info is not None:
            self.rollout_cache.history[-1].update(info)

        # Add new observation to history
        # Create new step history, including new observation returned by environment
        self.rollout_cache.history.append({
            "observation": observation,  # New observation (for next decision)
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,  # Remaining actions
            "messages": None  # Messages will be built in format_messages
        })
        
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        # In validation mode, save rendered frames (if environment supports)
        if self.mode == "val" and self.pipeline_config.render_save_dir and hasattr(self.env, "render"):
            frame = self.env.render(mode='rgb_array')
            if isinstance(frame, np.ndarray):
                self.rollout_cache.frames.append(frame)

        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        """
        Generate action using LLM.
        Formats messages, checks length limits, and calls LLM to generate response.
        """
        # Format messages, build model input
        lm_input = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]

        # Check if input sequence length exceeds limit
        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        # Calculate maximum number of tokens to generate
        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        
        # Prepare generation configuration, ensure max_new_tokens does not exceed total sequence length
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        
        # Set source rank information (for tracking in distributed environment)
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]

        # Collect all history messages (for LLM agent generation)
        # Extract all built messages from history, forming complete conversation context
        input_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        # Call LLM agent to generate response
        # Generate action/response through llm_proxy (possibly policy model, OpenAI API, or other LLM interface)
        # Note: In llm proxy inference, messages are not used, the effect is lm_input
        lm_output: DataProto = self.llm_proxy.generate(messages=input_messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)

        # Check if generation is successful
        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})

        # Extract and process generated response
        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()
        
        # Get current step's history content, for saving response information
        content = self.rollout_cache.history[-1]

        # Save log probabilities during inference (if exist)
        # These log probabilities are used for subsequent reinforcement learning training (computing policy gradients, etc.)
        if "infer_logprobs" in lm_output.batch:
            # Only take log probabilities of response part (excluding prompt part)
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
            content["infer_logprobs"] = infer_logprobs.tolist()

        # Save response information to history
        content["response_ids"] = response_ids
        
        # Decode response to text and add to messages, forming complete conversation record
        content["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})
        
        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: RolloutCache) -> DataProto:
        """
        Format messages into LLM input format.
        Builds message list, converts to token IDs, concatenates history, and creates input tensors.
        """
        # Get current step's content (latest step's history record)
        content = self.rollout_cache.history[-1]

        messages = []
        
        # If first step (step=0), add system prompt
        if self.rollout_cache.step == 0:
            # Add system message, including environment introduction and tool information
            messages.append({"role": "system", "content": self.agent_system_template})
        
        # Add observation as user content
        observation = content["observation"]
        if observation["type"] == "tool":
            # Qwen3 Tool Response Template Style
            messages.append({"role": "user", "content": f"<tool_response>\n{observation['content']}\n</tool_response>"})
        else:
            messages.append({"role": "user", "content": observation['content']})

        # Convert message list to token IDs (using chat template)
        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True, enable_thinking=True)
        print("enable_thinking: True")
        
        # Collect history conversation token IDs (previous all steps' prompt and response)
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])  # History steps' prompt token IDs
            history_token_ids.extend(items["response_ids"])  # History steps' response token IDs
        
        # If there is history conversation, add conversation end token before current prompt (for connecting history and new prompt)
        if len(history_token_ids):
            prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids
        
        # Concatenate history token IDs and current prompt token IDs, forming complete input sequence
        input_ids = history_token_ids + prompt_ids

        # Convert to PyTorch tensor format
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0) 
        position_ids = attention_mask.cumsum(dim=-1)  
        
        # Build model input data object
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        
        # Save current step's prompt_ids and messages to history, for subsequent steps
        content["prompt_ids"] = prompt_ids
        content["messages"] = messages
        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """
        Format complete episode trajectory for RL training.
        Concatenates tokens, computes rewards, creates masks, and pads to fixed length.
        Used for trajectory-wise training (e.g., StarPO).
        """
        # Clean history record
        # If last item contains observation (this is new observation added by step(), not used for decision), remove it
        if 'observation' in rollout_cache.history[-1]:
            rollout_cache.history.pop(-1)
        
        # Prepare history record (exclude last item)
        history = rollout_cache.history[:-1]
        # Copy last item and remove reward (because reward has been saved in previous step)
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        # Calculate reward
        scores = [i['reward'] for i in self.rollout_cache.history]
        # Calculate episode total reward (sum of all steps' rewards)
        episode_score = sum(scores)

        # Concatenate all steps' token sequences and generate masks
        token_ids = []
        prompt_masks = [] 
        response_masks = []
        infer_logprobs = []
        
        # Iterate over all history steps, concatenate token IDs and generate masks
        for items in self.rollout_cache.history:
            # Concatenate current step's prompt and response token IDs
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            
            # Generate prompt mask: prompt part is 1, response part is 0
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            # Generate response mask: prompt part is 0, response part is 1
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))
            
            # If there is inference log probability, also concatenate (prompt part is 0, response part is actual log probability)
            if "infer_logprobs" in items:
                infer_logprobs.extend([0] * len(items["prompt_ids"]) + items["infer_logprobs"])

        # Convert to PyTorch tensor format
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        # Generate prompt_mask and score_tensor
        # Find the position of the first response token
        first_response_idx = response_masks.index(1)
        # prompt_mask: from start to first response is 1, after first response is 0
        # This is used to distinguish prompt part and response part in the entire sequence
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask = torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        
        # Create score_tensor: place episode total reward at the last position of the sequence
        # This is the key for trajectory-wise RL training: place episode total reward at the last position of the sequence, for computing advantage function
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        
        # Position encoding
        position_ids = attention_mask.cumsum(dim=-1)

        # Create initial DataProto object
        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        # Calculate average response length (for metric statistics)
        response_length = response_mask.sum(dim=-1).float().mean().item()

        # Pad to fixed length (for batch processing)
        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        # Update batch data
        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,  # Used to mark which tokens need to calculate loss
            "prompt_mask": prompt_mask,  # Used to distinguish prompt and response
            "scores": score_tensor,  # episode total reward (for RL training)
        })
        
        # If there is inference log probability, also pad and add
        if len(infer_logprobs):
            infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
            infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
            # Remove log probability of the first token (usually is BOS token)
            lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

        # Add non-tensor metadata
        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "frames": np.array([self.rollout_cache.frames], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        # Aggregate environment metrics
        # Get metric aggregation mode (e.g., mean, last)
        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        # Collect all steps' metrics
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        # Aggregate metrics according to aggregation mode (e.g., action_is_valid取mean，success取last等)
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length  # 添加响应长度指标
        lm_input.meta_info = {"metrics": env_metric}
        return lm_input