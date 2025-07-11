# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
RLVR DataParallel PPO Actor
"""

import logging
import os
from collections import defaultdict

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, compute_rlvr_policy_loss, kl_penalty
from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RLVRDataParallelPPOActor(DataParallelPPOActor):
    """
    DataParallelPPOActor for RLVR algorithm.
    It overrides the update_policy method to include a high-entropy token mask in the loss calculation.
    """

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        metrics = defaultdict(list)
        
        # RLVR: Calculate high entropy mask based on the old policy's entropies for the whole batch
        high_entropy_percentile = self.config.get("high_entropy_percentile", 0.0)
        high_entropy_mask = None
        if "old_entropys" in data.batch and high_entropy_percentile > 0:
            old_entropys = data.batch["old_entropys"]
            response_mask = data.batch["response_mask"]
            
            # Flatten entropies of valid response tokens
            if response_mask.any():
                valid_entropies = old_entropys[response_mask.bool()]
                if valid_entropies.numel() > 0:
                    threshold = torch.quantile(valid_entropies.float(), high_entropy_percentile)
                    high_entropy_mask = (old_entropys >= threshold).float()
                    metrics["actor/entropy_threshold"].append(threshold.item())


        for ppo_epoch in range(self.config.ppo_epochs):
            # A new dataloader for each ppo epoch
            mini_batch_data_list, _ = self._get_micro_batches(data)

            # iterate over all data
            for i, mini_batch_data in enumerate(mini_batch_data_list):
                # zero grad
                self.actor_optimizer.zero_grad()
                mini_batch_metrics = defaultdict(list)

                # iterate over all micro batches
                for micro_batch_data in mini_batch_data:
                    old_log_prob = micro_batch_data["old_log_prob"]
                    advantages = micro_batch_data["advantages"]
                    response_mask = micro_batch_data["response_mask"]

                    # RLVR: Get the corresponding slice of the high entropy mask
                    micro_high_entropy_mask = None
                    if high_entropy_mask is not None:
                        start_idx = micro_batch_data.meta_info['start_idx']
                        end_idx = micro_batch_data.meta_info['end_idx']
                        micro_high_entropy_mask = high_entropy_mask[start_idx:end_idx]

                    entropy_coeff = self.config.get("entropy_coeff", 0.0)
                    calculate_entropy = entropy_coeff != 0

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch_data.batch,
                        temperature=self.config.get("temperature", 1.0),
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = log_prob.to(torch.float32)

                    # compute policy loss
                    loss_agg_mode = self.config.loss_agg_mode
                    
                    # RLVR: Use the new policy loss function
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_rlvr_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        high_entropy_mask=micro_high_entropy_mask,
                        cliprange=self.config.clip_ratio,
                        cliprange_low=self.config.clip_ratio_low,
                        cliprange_high=self.config.clip_ratio_high,
                        clip_ratio_c=self.config.clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_batch_data["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(mat=kld, mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"].append(kl_loss.detach().item())

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(micro_batch_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"].append(pg_loss.detach().item())
                    micro_batch_metrics["actor/pg_clipfrac"].append(pg_clipfrac.detach().item())
                    micro_batch_metrics["actor/ppo_kl"].append(ppo_kl.detach().item())
                    micro_batch_metrics["actor/pg_clipfrac_lower"].append(pg_clipfrac_lower.detach().item())

                # grad clip and optimizer step
                grad_norm = self._optimizer_step()
                mini_batch_metrics["actor/grad_norm"] = [grad_norm.detach().item()]
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics 