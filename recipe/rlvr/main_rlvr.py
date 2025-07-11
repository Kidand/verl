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
Main entrypoint for RLVR training.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.trainer.ppo.ray_trainer import RayTrainer # Use the original RayTrainer
from verl.utils.device import is_cuda_available


@hydra.main(config_path="../dapo/config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_rlvr(config)


def run_rlvr(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )
    
    TaskRunner.remote().run.remote(config)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        
        # --- MONKEY PATCHING ---
        # We replace the default actor class with our custom RLVR actor class
        # before any workers that use it are instantiated. This is the correct
        # way to inject custom logic without modifying the core library.
        import verl.workers.actor.dp_actor
        from verl.workers.actor.rlvr_dp_actor import RLVRDataParallelPPOActor
        verl.workers.actor.dp_actor.DataParallelPPOActor = RLVRDataParallelPPOActor
        # --- END MONKEY PATCHING ---

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        # Now, when ActorRolloutRefWorker is imported and used, it will internally
        # instantiate our RLVRDataParallelPPOActor instead of the default one.
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        # Correct role worker mapping (removed Role.Actor)
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {role: global_pool_id for role in role_worker_mapping}

        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        from verl.workers.reward_manager import get_reward_manager_cls
        reward_manager_cls = get_reward_manager_cls(config.reward_model.get("reward_manager", "naive"))
        
        compute_score = get_custom_reward_fn(config)
        common_reward_args = {
            "tokenizer": tokenizer,
            "compute_score": compute_score,
            "reward_fn_key": config.data.reward_fn_key,
            "max_resp_len": config.data.max_response_length,
            "overlong_buffer_cfg": config.reward_model.overlong_buffer,
        }
        reward_fn = reward_manager_cls(**common_reward_args, num_examine=0)
        val_reward_fn = reward_manager_cls(**common_reward_args, num_examine=1)
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # RLVR: Use the original RayTrainer, the monkey patching above is sufficient.
        trainer = RayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main() 