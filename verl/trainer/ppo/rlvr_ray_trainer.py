# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
RLVR Ray Trainer
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


class RLVRRayTrainer(RayPPOTrainer):
    """
    Ray Trainer for RLVR. Inherits from DAPO trainer and modifies the data handling
    to keep entropies from the old policy for loss masking.
    """

    def fit(self):
        """
        The training loop of RLVR.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # Simplified pop
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"] + (["multi_modal_data"] if "multi_modal_data" in new_batch.non_tensor_batch else []),
                )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # Simplified reward calculation
                        reward_result = self.reward_fn(new_batch, return_dict=True)
                        new_batch.batch["token_level_scores"] = reward_result["reward_tensor"]
                        if "reward_extra_info" in reward_result:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_result["reward_extra_info"].items()})
                        
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name not in new_batch.non_tensor_batch:
                             new_batch.non_tensor_batch[metric_name] = new_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        
                        kept_prompt_uids = [uid for uid, vals in prompt_uid2metric_vals.items() if np.std(vals) > 0 or len(vals) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)
                        
                        kept_traj_idxs = [i for i, uid in enumerate(new_batch.non_tensor_batch["uid"]) if uid in kept_prompt_uids]
                        
                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            # Handle batch accumulation if needed
                            continue
                    
                # The rest of the training step starts here
                
                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, "blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch, calculate_entropy=True)
                    
                    # RLVR change: keep entropies from the old policy
                    batch.batch["old_entropys"] = old_log_prob.batch.pop("entropys")
                    
                    entropys = batch.batch["old_entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics.update({"actor/entropy": entropy_agg.detach().item()})
                    
                    batch = batch.union(old_log_prob)

                # The rest of the PPO step (adv calculation, actor/critic update) follows
                # ... (This part is identical to RayDAPOTrainer)
                with marked_timer("adv", timing_raw, "brown"):
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    )

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Reset for next batch accumulation
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                
                # Logging, saving, validation
                progress_bar.update(1)
                self.global_steps += 1
                if self.global_steps > self.total_training_steps:
                    break
            if self.global_steps > self.total_training_steps:
                break 