import functools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets.distributed
import torch
import torch.backends
import wandb
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_layerwise_casting
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm

from timecolor import data, logging, models, optimizer, parallel, patches, utils
from timecolor.args import BaseArgsType
from timecolor.config import TrainingType
from timecolor.state import State, TrainState

from ..base import Trainer
from .config import SFTFullRankConfig, SFTLowRankConfig


ArgsType = Union[BaseArgsType, SFTFullRankConfig, SFTLowRankConfig]

logger = logging.get_logger()

def unescape_caption(encoded: str) -> str:
    """Decode back to the original caption (for later reading)."""
    return json.loads(f'"{encoded}"')

class SFTTrainer(Trainer):
    def __init__(self, args: ArgsType, model_specification: models.ModelSpecification) -> None:
        super().__init__(args)

        self.state = State()
        self.state.train_state = TrainState()

        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None

        # Text encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None

        # Image encoders
        self.image_encoder = None
        self.image_processor = None

        # Denoisers
        self.transformer = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        # Optimizer & LR scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # Checkpoint manager
        self.checkpointer = None

        self._init_distributed()
        self._init_config_options()

        # Perform any patches that might be necessary for training to work as expected
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)

        self.model_specification = model_specification
        self._are_condition_models_loaded = False

    #TODO (Bryan): Count trainable parameters
    def count_trainable_params_and_bytes(self, *modules: torch.nn.Module):
        """
        Returns (num_params, num_bytes) for all parameters that require grad
        across the supplied modules.
        """
        num_params = 0
        num_bytes  = 0
        for m in modules:
            for p in m.parameters():
                if p.requires_grad:
                    n = p.numel()
                    num_params += n
                    num_bytes  += n * p.element_size()   # element_size() → 2 for fp16, 4 for fp32, etc.
        return num_params, num_bytes

    def run(self) -> None:
        try:
            self._prepare_models()
            self._prepare_trainable_parameters()
            self._prepare_for_training()
            n_params, n_bytes = self.count_trainable_params_and_bytes(self.transformer)
            
            logger.info(
                f"✨ Trainable parameters: {n_params:,} "
                f"({n_params/1e6:.2f} M)  |  Memory ≈ {n_bytes/1024**2:.2f} MB "
                f"({n_bytes/1024**3:.2f} GB)"
            )

            logger.info("Starting preparing dataset")
            self._prepare_dataset()
            logger.info("Preparig dataset finished")
            self._prepare_checkpointing()
            logger.info("Starting checkpointing dataset")
            self._train()
            # trainer._evaluate()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def eval(self) -> None:
        try:
            #self._prepare_models()
            #self._prepare_trainable_parameters()
            #self._prepare_for_training()
            #self._prepare_dataset()
            #self._prepare_checkpointing()
            self._eval()
        except Exception as e:
            logger.error(f"Error during eval: {e}")
            self.state.parallel_backend.destroy()
            raise e

    def _prepare_models(self) -> None:
        logger.info("Initializing models")
        if self.args.custom_transformer_path is not None:
            logger.info("Entering custom transformer path!")
            logger.info(f"Loading custom transformer from {self.args.custom_transformer_path}")
        diffusion_components = self.model_specification.load_diffusion_models(custom_path=self.args.custom_transformer_path)
        self._set_components(diffusion_components)

        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def _prepare_trainable_parameters(self) -> None:
        logger.info("Initializing trainable parameters")

        parallel_backend = self.state.parallel_backend

        if self.args.training_type == TrainingType.FULL_FINETUNE or self.args.training_type == TrainingType.TEMPORAL_FULL_FINETUNE:
            logger.info("Finetuning transformer with no additional parameters")
            utils.set_requires_grad([self.transformer], True)
        else:
            logger.info("Finetuning transformer with PEFT parameters")
            utils.set_requires_grad([self.transformer], False)

        # Layerwise upcasting must be applied before adding the LoRA adapter.
        # If we don't perform this before moving to device, we might OOM on the GPU. So, best to do it on
        # CPU for now, before support is added in Diffusers for loading and enabling layerwise upcasting directly.
        if (self.args.training_type == TrainingType.LORA or \
            self.args.training_type == TrainingType.TEMPORAL_LORA) and "transformer" in self.args.layerwise_upcasting_modules:
            apply_layerwise_casting(
                self.transformer,
                storage_dtype=self.args.layerwise_upcasting_storage_dtype,
                compute_dtype=self.args.transformer_dtype,
                skip_modules_pattern=self.args.layerwise_upcasting_skip_modules_pattern,
                non_blocking=True,
            )

        transformer_lora_config = None
        if (self.args.training_type == TrainingType.LORA or \
            self.args.training_type == TrainingType.TEMPORAL_LORA):
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)

        # Make sure the trainable params are in float32 if data sharding is not enabled. For FSDP, we need all
        # parameters to be of the same dtype.
        if parallel_backend.data_sharding_enabled:
            self.transformer.to(dtype=self.args.transformer_dtype)
        else:
            if self.args.training_type == TrainingType.LORA:
                cast_training_params([self.transformer], dtype=torch.float32)

    def _prepare_for_training(self) -> None:
        # 1. Apply parallelism
        parallel_backend = self.state.parallel_backend
        model_specification = self.model_specification

        if parallel_backend.context_parallel_enabled:
            raise NotImplementedError(
                "Context parallelism is not supported yet. This will be supported in the future."
            )

        if parallel_backend.tensor_parallel_enabled:
            # TODO(aryan): handle fp8 from TorchAO here
            model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh("tp"),
                transformer=self.transformer,
            )

        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            # TODO(aryan): support other checkpointing types
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")

        # Apply torch.compile
        self._maybe_torch_compile()

        # Enable DDP, FSDP or HSDP
        if parallel_backend.data_sharding_enabled:
            # TODO(aryan): remove this when supported
            if self.args.parallel_backend == "accelerate":
                raise NotImplementedError("Data sharding is not supported with Accelerate yet.")

            if parallel_backend.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")

            # Apply FSDP or HSDP
            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)

            parallel_backend.apply_fsdp2(
                model=self.transformer,
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,  # TODO(aryan): needs to be tested and allowed for enabling later
                device_mesh=parallel_backend.get_mesh()[dp_mesh_names],
            )
        elif parallel_backend.data_replication_enabled:
            logger.info("Applying DDP to the model")

            if parallel_backend.get_mesh().ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")

            parallel_backend.apply_ddp(self.transformer, parallel_backend.get_mesh())
        else:
            parallel_backend.prepare_model(self.transformer)

        self._move_components_to_device()

        # 2. Prepare optimizer and lr scheduler
        # For training LoRAs, we can be a little more optimal. Currently, the OptimizerWrapper only accepts torch::nn::Module.
        # This causes us to loop over all the parameters (even ones that don't require gradients, as in LoRA) at each optimizer
        # step. This is OK (see https://github.com/pytorch/pytorch/blob/2f40f789dafeaa62c4e4b90dbf4a900ff6da2ca4/torch/optim/sgd.py#L85-L99)
        # but can be optimized a bit by maybe creating a simple wrapper module encompassing the actual parameters that require
        # gradients. TODO(aryan): look into it in the future.
        model_parts = [self.transformer]
        self.state.num_trainable_parameters = sum(
            p.numel() for m in model_parts for p in m.parameters() if p.requires_grad
        )

        # Setup distributed optimizer and lr scheduler
        logger.info("Initializing optimizer and lr scheduler")
        self.state.train_state = TrainState()
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
            # TODO(aryan): handle last_epoch
        )
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(self.optimizer, self.lr_scheduler)

        # 3. Initialize trackers, directories and repositories
        self._init_logging()
        self._init_trackers()
        self._init_directories_and_repositories()

    def _prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")

        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            latent_base_path = config.pop("latent_base_path", None)
            stage2_latent_base_path = config.pop("stage2_latent_base_path", None)
            stage2_latent_recolor_base_path = config.pop("stage2_latent_recolor_base_path", None)
            caption_options = config.pop("caption_options", {})

            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")

            dataset_name_or_root = data_root or dataset_file
            print("Add this stage use recolor mask is: ", self.args.use_recolor_mask)
            dataset = data.initialize_dataset(
                dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options, 
                latent_base_path = latent_base_path, 
                stage2_latent_base_path=stage2_latent_base_path,
                stage2_latent_recolor_base_path= stage2_latent_recolor_base_path,
                use_recolor_mask=self.args.use_recolor_mask
            )

            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(
                    f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
                )

            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)

        if self.args.combine_multistage_datasets_with_ratio:
            #TODO (Bryan): there are multiple dataset, separate with two variables to not destroy the 
            assert len(datasets) == len(self.args.combine_multistage_datasets_with_ratio), f"len of datasets: {len(datasets)}, len of args.combine_multistage_datasets_with_ratio: {len(self.args.combine_multistage_datasets_with_ratio)}"
            dataset_list = []
            dataloader_list = []
            for ds in datasets:
                dataset_list.append(data.combine_datasets([ds], buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True))
                dataloader_list.append(self.state.parallel_backend.prepare_dataloader(
                    dataset_list[-1], batch_size=1, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.pin_memory
                ))
            self.dataset = None
            self.dataloader = None
            self.dataset_list = dataset_list
            self.dataloader_list = dataloader_list

        else:
            dataset = data.combine_datasets(datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
            dataloader = self.state.parallel_backend.prepare_dataloader(
                dataset, batch_size=1, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.pin_memory
            )
            self.dataset = dataset
            self.dataloader = dataloader
            self.dataset_list = None
            self.dataloader_list = None

    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.LORA or self.args.training_type == TrainingType.TEMPORAL_LORA:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
                    self.model_specification._save_lora_weights(
                        os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}"),
                        state_dict,
                        self.scheduler,
                        metadata,
                    )
                elif self.args.training_type == TrainingType.FULL_FINETUNE or self.args.training_type == TrainingType.TEMPORAL_FULL_FINETUNE:
                    self.model_specification._save_model(
                        os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"),
                        self.transformer,
                        state_dict,
                        self.scheduler,
                    )
                else:
                    raise ValueError(
                        f"Unsupported training type {self.args.training_type} for saving model weights."
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        self.checkpointer = parallel_backend.get_checkpointer(
            dataloader=self.dataloader,
            model_parts=[self.transformer],
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            dataloader_list=self.dataloader_list,
            _callback_fn=save_model_hook,
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            print("Just beginning new stage:", self.args.just_beginning_new_stage)
            self.checkpointer.load(resume_from_checkpoint, just_beginning_new_stage=self.args.just_beginning_new_stage)

    def _train(self) -> None:
        logger.info("Starting training")

        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        device = parallel_backend.device
        dtype = self.args.transformer_dtype

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        global_batch_size = self.args.batch_size * parallel_backend._dp_degree
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "train steps": self.args.train_steps,
            "per-replica batch size": self.args.batch_size,
            "global batch size": global_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=train_state.step,
            desc="Training steps",
            disable=not parallel_backend.is_local_main_process,
        )

        generator = torch.Generator(device=device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        patch_size = 1
        if (
            getattr(self.transformer.config, "patch_size", None) is not None
            and getattr(self.transformer.config, "patch_size_t", None) is not None
        ):
            patch_size = self.transformer.config.patch_size * self.transformer.config.patch_size_t
        elif isinstance(getattr(self.transformer.config, "patch_size", None), int):
            patch_size = self.transformer.config.patch_size
        elif isinstance(getattr(self.transformer.config, "patch_size", None), (list, tuple)):
            patch_size = math.prod(self.transformer.config.patch_size)

        scheduler_sigmas = utils.get_scheduler_sigmas(self.scheduler)
        scheduler_sigmas = (
            scheduler_sigmas.to(device=device, dtype=torch.float32) if scheduler_sigmas is not None else None
        )
        scheduler_alphas = utils.get_scheduler_alphas(self.scheduler)
        scheduler_alphas = (
            scheduler_alphas.to(device=device, dtype=torch.float32) if scheduler_alphas is not None else None
        )
        timesteps_buffer = []

        self.transformer.train()
        
        #if data loader is a list, data_iterator should be a list as well
        #
        if isinstance(self.dataloader_list, list):
            print("Entering list data loader")
            data_iterator_list = [iter(dl) for dl in self.dataloader_list]
        else:
            print("Entering normal data loader")
            data_iterator = iter(self.dataloader)

        #TODO(Bryan): Do not hardcode this
        compute_posterior = False
        assert self.args.enable_precomputation
        #assert self.args.precomputation_reuse
        #compute_posterior = False if self.args.enable_precomputation else (not self.args.precomputation_once)
        

        #TODO(Bryan): Integrate the preencoded dataset later on
        # preprocessor = data.initialize_preprocessor(
        #     rank=parallel_backend.rank,
        #     world_size=parallel_backend.world_size,
        #     num_items=self.args.precomputation_items if self.args.enable_precomputation else 1,
        #     processor_fn={
        #         #TODO(Bryan): Do not hardcode this
        #         "condition": self.model_specification.prepare_conditions_preencoded,
        #         "latent": functools.partial(
        #             self.model_specification.prepare_latents_preencoded, compute_posterior=compute_posterior
        #         ),
        #     },
        #     #TODO(Bryan): Pay attention to this part
        #     save_dir=self.args.precomputation_dir,
        #     enable_precomputation=self.args.enable_precomputation,
        #     enable_reuse=self.args.precomputation_reuse,
        # )
        # condition_iterator: Iterable[Dict[str, Any]] = None
        # latent_iterator: Iterable[Dict[str, Any]] = None
        # sampler = data.ResolutionSampler(
        #     batch_size=self.args.batch_size, dim_keys=self.model_specification._resolution_dim_keys
        # )
        requires_gradient_step = True
        accumulated_loss = 0.0
        cur_total = None
        combine_multistage_datasets_accumulated = None
        if self.args.combine_multistage_datasets_with_ratio:
            cur_total = sum(self.args.combine_multistage_datasets_with_ratio)
            #accumulated array = 
            combine_multistage_datasets_accumulated = [self.args.combine_multistage_datasets_with_ratio[0]]
            for i in range(1, len(self.args.combine_multistage_datasets_with_ratio)):
                combine_multistage_datasets_accumulated.append(combine_multistage_datasets_accumulated[-1] + self.args.combine_multistage_datasets_with_ratio[i])
            print("entering multistage dataset")
            print("combine_multistage_datasets_accumulated: ", combine_multistage_datasets_accumulated)

        #TODO(Bryan): Erase this, this is my mistake
        # print("Reach here 1")
        # for i in range(len(data_iterator_list)):
        #     for _ in tqdm(range(2500)):
        #         next(data_iterator_list[i])
        # train_state.step += 5000
        # parallel_backend.wait_for_everyone()
        # print()
        # print("Reach here 2")

        while (
            train_state.step < self.args.train_steps and train_state.observed_data_samples < self.args.max_data_samples
        ):
            # 1. Load & preprocess data if required
            # TODO(Bryan): Do not hardcode this
            # if preprocessor.requires_data:
            #     condition_iterator, latent_iterator = self._prepare_data(preprocessor, data_iterator)
            # # 2. Prepare batch
            # try:
            #     #TODO(Bryan): Do not hardcode this, for now because everything is preencoded I dont need preprocessor
            #     condition_item = next(condition_iterator)
            #     latent_item = next(latent_iterator)
            #     #TODO(Bryan): Do not hardcode this
            #     #sampler.consume(condition_item, latent_item)
            # except StopIteration:
            #     if requires_gradient_step:
            #         self.optimizer.step()
            #         self.lr_scheduler.step()
            #         requires_gradient_step = False
            #     logger.info("Data exhausted. Exiting training loop.")
            #     break
            # except Exception as e:
            #     print("Some operation enter other error")
            #     print(e)
            #     raise e
            # if sampler.is_ready:
            #     condition_batch, latent_batch = sampler.get_batch()
            #     condition_model_conditions = self.model_specification.collate_conditions(condition_batch)
            #     latent_model_conditions = self.model_specification.collate_latents(latent_batch)
            # else:
            #     continue


            #TODO(Bryan): Mitigate using this later

            #force skip because of my mistake
            try:
                #TODO(Bryan): Do not hardcode this, for now because everything is preencoded I dont need preprocessor
                if isinstance(self.dataloader_list, list):
                    assert cur_total is not None, "cur_total should not be None when using multiple dataloaders"
                    cur_turn = train_state.step % cur_total
                    #look which index to use
                    index = 0
                    for i, ratio in enumerate(combine_multistage_datasets_accumulated):
                        if cur_turn < ratio:
                            index = i
                            break
                    next_data_piece = next(data_iterator_list[index])
                    cur_stage = index + 1
                else:
                    next_data_piece = next(data_iterator)
                    cur_stage = self.args.stage
                condition_model_conditions = self.model_specification.prepare_conditions_preencoded(**next_data_piece, generator=self.state.generator, stage=cur_stage)
                latent_model_conditions = self.model_specification.prepare_latents_preencoded(
                    **next_data_piece, generator=self.state.generator, compute_posterior=compute_posterior, stage=cur_stage
                )
            except StopIteration:
                if requires_gradient_step:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    requires_gradient_step = False
                logger.info("Data exhausted. Exiting training loop.")
                break
            except Exception as e:
                print("Some operation enter other error")
                print(e)
                raise e

            train_state.step += 1
            train_state.observed_data_samples += self.args.batch_size * parallel_backend._dp_degree

            lmc_latents = latent_model_conditions["latents"]
            # TODO(aryan): observed_num_tokens this needs to be allreduced
            train_state.observed_num_tokens += math.prod(lmc_latents.shape[:-1]) // patch_size

            logger.debug(f"Starting training step ({train_state.step}/{self.args.train_steps})")

            latent_model_conditions = utils.align_device_and_dtype(latent_model_conditions, device, dtype)
            condition_model_conditions = utils.align_device_and_dtype(condition_model_conditions, device, dtype)
            latent_model_conditions = utils.make_contiguous(latent_model_conditions)
            condition_model_conditions = utils.make_contiguous(condition_model_conditions)

            # 3. Forward pass
            sigmas = utils.prepare_sigmas(
                scheduler=self.scheduler,
                sigmas=scheduler_sigmas,
                batch_size=self.args.batch_size,
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                flow_weighting_scheme=self.args.flow_weighting_scheme,
                flow_logit_mean=self.args.flow_logit_mean,
                flow_logit_std=self.args.flow_logit_std,
                flow_mode_scale=self.args.flow_mode_scale,
                device=device,
                generator=self.state.generator,
            )
            sigmas = utils.expand_tensor_dims(sigmas, latent_model_conditions["latents"].ndim)
            parallel_backend.wait_for_everyone()
            # NOTE: for planned refactor, make sure that forward and backward pass run under the context.
            # If only forward runs under context, backward will most likely fail when using activation checkpointing
            assert cur_stage != -1, "Stage must be set for training"
            assert self.args.rope_enumeration_method != "", "ROPE enumeration method must be set for training"


            #TODO(Bryan): Temporarily disable the attention provider context to see if it works
            with self.attention_provider_ctx(training=True):
                pred, target, sigmas = self.model_specification.forward(
                    transformer=self.transformer,
                    scheduler=self.scheduler,
                    condition_model_conditions=condition_model_conditions,
                    latent_model_conditions=latent_model_conditions,
                    sigmas=sigmas,
                    generator=self.state.generator,
                    compute_posterior=compute_posterior,
                    #(BRYAN): DO NOT HARDCODE THIS
                    stage=cur_stage,
                    rope_enumeration_method=self.args.rope_enumeration_method,
                    no_mask_ablation = self.args.no_mask_ablation
                )

                timesteps = (sigmas * 1000.0).long()
                weights = utils.prepare_loss_weights(
                    scheduler=self.scheduler,
                    alphas=scheduler_alphas[timesteps] if scheduler_alphas is not None else None,
                    sigmas=sigmas,
                    flow_weighting_scheme=self.args.flow_weighting_scheme,
                )
                weights = utils.expand_tensor_dims(weights, pred.ndim)

                # 4. Compute loss & backward pass
                loss = weights.float() * (pred.float() - target.float()).pow(2)
                # Average loss across all but batch dimension
                loss = loss.mean(list(range(1, loss.ndim)))
                # Average loss across batch dimension
                loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.detach().item()
                requires_gradient_step = True

            # 5. Clip gradients
            model_parts = [self.transformer]
            grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.max_grad_norm,
                foreach=True,
                pp_mesh=parallel_backend.get_mesh("pp") if parallel_backend.pipeline_parallel_enabled else None,
            )

            # 6. Step optimizer & log metrics
            logs = {}

            if train_state.step % self.args.gradient_accumulation_steps == 0:
                # TODO(aryan): revisit no_sync() for FSDP
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
                if (
                    parallel_backend.data_replication_enabled
                    or parallel_backend.data_sharding_enabled
                    or parallel_backend.context_parallel_enabled
                ):
                    dp_cp_mesh = parallel_backend.get_mesh("dp_cp")
                    global_avg_loss, global_max_loss = (
                        parallel.dist_mean(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                        parallel.dist_max(torch.tensor([accumulated_loss], device=device), dp_cp_mesh),
                    )
                else:
                    global_avg_loss = global_max_loss = accumulated_loss

                logs["global_avg_loss"] = global_avg_loss
                logs["global_max_loss"] = global_max_loss
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                accumulated_loss = 0.0
                requires_gradient_step = False

            progress_bar.update(1)
            progress_bar.set_postfix(logs)

            timesteps_buffer.extend([(train_state.step, t) for t in timesteps.detach().cpu().numpy().tolist()])

            if train_state.step % self.args.logging_steps == 0:
                # TODO(aryan): handle non-SchedulerWrapper schedulers (probably not required eventually) since they might not be dicts
                # TODO(aryan): causes NCCL hang for some reason. look into later
                # logs.update(self.lr_scheduler.get_last_lr())

                # timesteps_table = wandb.Table(data=timesteps_buffer, columns=["step", "timesteps"])
                # logs["timesteps"] = wandb.plot.scatter(
                #     timesteps_table, "step", "timesteps", title="Timesteps distribution"
                # )
                timesteps_buffer = []

                logs["observed_data_samples"] = train_state.observed_data_samples
                logs["observed_num_tokens"] = train_state.observed_num_tokens

                parallel_backend.log(logs, step=train_state.step)
                train_state.log_steps.append(train_state.step)

            # 7. Save checkpoint if required
            self.checkpointer.save(
                step=train_state.step, _device=device, _is_main_process=parallel_backend.is_main_process
            )
            parallel_backend.wait_for_everyone()

            # 8. Perform validation if required
            if not self.args.disable_validation and train_state.step % self.args.validation_steps == 0:
                self._validate(step=train_state.step, final_validation=False)

        # 9. Final checkpoint, validation & cleanup
        self.checkpointer.save(
            train_state.step, force=True, _device=device, _is_main_process=parallel_backend.is_main_process
        )
        parallel_backend.wait_for_everyone()
        if not self.args.disable_validation:
            #TODO (Bryan): Fix the bug with Final Validation: OOM
            self._validate(step=train_state.step, final_validation=False)

        self._delete_components()
        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        # 10. Upload artifacts to hub
        if parallel_backend.is_main_process and self.args.push_to_hub:
            upload_folder(
                repo_id=self.state.repo_id,
                folder_path=self.args.output_dir,
                ignore_patterns=[f"{self.checkpointer._prefix}_*"],
            )

        parallel_backend.destroy()

    def _validate(self, step: int, final_validation: bool = False, run_from_eval: bool = False) -> None:
        if self.args.validation_dataset_file is None:
            return

        logger.info("Starting validation")
        logger.info("Step: ")
        logger.info(step)
        logger.info("Is Final Validation")
        logger.info(final_validation)
        logger.info("Run from eval")
        logger.info(run_from_eval)

        # 1. Load validation dataset
        parallel_backend = self.state.parallel_backend

        # Hack to make accelerate work. TODO(aryan): refactor
        dp_mesh = None
        if parallel_backend.world_size > 1:
            dp_mesh = parallel_backend.get_mesh("dp_replicate")

        if dp_mesh is not None:
            local_rank, dp_world_size = dp_mesh.get_local_rank(), dp_mesh.size()
        else:
            local_rank, dp_world_size = 0, 1

        #TODO (Bryan): Change this to your required format
        if self.args.stage == 0:
            dataset = data.ValidationDatasetMultisubjectZeroStage(self.args.validation_dataset_file)
        elif self.args.stage == 1:
            dataset = data.ValidationDatasetMultisubjectFirstStage(self.args.validation_dataset_file)
        elif self.args.stage == 2:
            dataset = data.ValidationDatasetMultisubjectSecondStage(self.args.validation_dataset_file)
        else:
            raise ValueError(f"Unsupported stage {self.args.stage} for validation dataset.")
        dataset._data = datasets.distributed.split_dataset_by_node(dataset._data, local_rank, dp_world_size)
        validation_dataloader = data.DPDataLoader(
            local_rank,
            dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items,
        )
        data_iterator = iter(validation_dataloader)
        main_process_prompts_to_filenames = {}  # Used to save model card
        all_processes_artifacts = []  # Used to gather artifacts from all processes

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        seed = self.args.seed if self.args.seed is not None else 0
        generator = torch.Generator(device=parallel_backend.device).manual_seed(seed)
        pipeline = self._init_pipeline(final_validation=final_validation)

        # 2. Run validation
        # TODO(aryan): when running validation with FSDP, if the number of data points is not divisible by dp_shards, we
        # will hang indefinitely. Either pad the dataset or raise an error early on during initialization if the dataset
        # size is not divisible by dp_shards.
        self.transformer.eval()
        while True:
            validation_data = next(data_iterator, None)
            if validation_data is None:
                break

            validation_data = validation_data[0]
            CUSTOM_FILE_NAME = validation_data.get("custom_output_name", None)
            print("Custom file name: ", CUSTOM_FILE_NAME)
            SKETCH_CONTROL_VIDEO = validation_data["sketch_control"]
            EXPORT_FPS = validation_data["frame_rate"]
            ORIGINAL_SKETCH_PATH = validation_data["granular_sketch_control_path"]            
            #TODO (Bryan): Check the key is it export_fps or frame_rate
            #PROMPT IS IN JSON ESCAPE, SO YOU NEED TO UNESCAPE IT
            #validation_data["prompt"] = unescape_caption(validation_data["prompt"])
            PROMPT = validation_data["prompt"]
            caption = PROMPT
            if self.args.stage == 0:
                STARTING_FULL_FRAME_REF_IMAGE = validation_data["starting_full_frame_ref"]
            elif self.args.stage == 1:
                FULL_FRAME_REF_IMAGE = validation_data["full_frame_ref"]
            elif self.args.stage == 2:
                FINAL_17CORRESPONDENCE_VIDEO = validation_data["final_17correspondence_video"]
                REFERENCES = validation_data["references"]
            else:
                raise ValueError(f"Unsupported stage {self.args.stage} for validation dataset.")
            print("Original Sketch Path: ", ORIGINAL_SKETCH_PATH)

            
            filename = "validation-" if not final_validation else "final-"
            #TODO(Bryan): Custom image names
            if CUSTOM_FILE_NAME:
                filename += CUSTOM_FILE_NAME
            else:
                filename += f"{step}-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.mp4"
            #filename += f"{step}-{rank}-{index}-{prompt_filename}-{time_}.{ext}"
            output_filename = os.path.join(self.args.output_dir, filename)

            
            #skip if file has existed
            if os.path.exists(output_filename):
                logger.info("Skipping existing file: ")
                logger.info(str(output_filename))
                continue


            with self.attention_provider_ctx(training=False):
                validation_artifacts = self.model_specification.validation(
                    pipeline=pipeline, generator=generator,
                    stage=self.args.stage,
                    rope_enumeration_method=self.args.rope_enumeration_method,
                    no_mask_ablation=self.args.no_mask_ablation,
                    force_every_validation_to_be_stage2_latent=self.args.force_every_validation_to_be_stage2_latent,
                    **validation_data,
                )

            # 2.1. If there are any initial images or videos, they will be logged to keep track of them as
            # conditioning for generation.
            prompt_filename = utils.string_to_filename(PROMPT)[:25]
            if self.args.stage == 0:
                artifacts = {
                    "starting_full_frame_ref": data.ImageArtifact(value=STARTING_FULL_FRAME_REF_IMAGE),
                    "sketch_control": data.VideoArtifact(value=SKETCH_CONTROL_VIDEO),
                }
            elif self.args.stage == 1:
                artifacts = {
                    "full_frame_ref": data.ImageArtifact(value=FULL_FRAME_REF_IMAGE),
                    "sketch_control": data.VideoArtifact(value=SKETCH_CONTROL_VIDEO),
                }
            elif self.args.stage == 2:
                #print("Final 17 Correspondence Video: ", FINAL_17CORRESPONDENCE_VIDEO)
                #print("Sketch control: ", SKETCH_CONTROL_VIDEO)
                artifacts = {
                    "final_17correspondence_video": data.VideoArtifact(value=FINAL_17CORRESPONDENCE_VIDEO),
                    "references": [data.ImageArtifact(value=ref) for ref in REFERENCES],
                    "sketch_control": data.VideoArtifact(value=SKETCH_CONTROL_VIDEO),
                }


            # 2.2. Track the artifacts generated from validation
            for i, validation_artifact in enumerate(validation_artifacts):
                if validation_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": validation_artifact})
            #print("Beginning ")
            # 2.3. Save the artifacts to the output directory and create appropriate logging objects
            # TODO(aryan): Currently, we only support WandB so we've hardcoded it here. Needs to be revisited.
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact, list))
                if isinstance(artifact, list):
                    # If the artifact is a list, we assume it is a list of ImageArtifacts
                    #print("reach here1")
                    assert all(isinstance(a, data.ImageArtifact) for a in artifact), "All artifacts in the list must be ImageArtifact"
                    ext = artifact[0].file_extension
                elif artifact.value is None:
                    #print("reach here2")
                    continue
                else:
                    try:
                        #print("reach here3")
                        #print("Artifact key: ", key)
                        ext = artifact.file_extension
                    except AttributeError:
                        print("Artifact value does not have file_extension, the value of artifact value is: ", artifact.value)
                        raise ValueError("Artifact value must have file_extension attribute")

                time_, rank = int(time.time()), parallel_backend.rank
                
                print("Filename before: ", filename)

                if self.args.stage == 0 and key == "starting_full_frame_ref":
                    continue
                    filename = f"starting_full_frame_ref-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.{ext}"
                    caption = f"[starting_full_frame_ref] {caption}"
                if self.args.stage == 1 and key == "full_frame_ref":
                    continue
                    filename = f"full_frame_ref-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.{ext}"
                    caption = f"[full_frame_ref] {caption}"
                elif self.args.stage == 2 and key == "final_17correspondence_video":
                    continue
                    filename = f"final_17correspondence_video-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.{ext}"
                    caption = f"[final_17correspondence_video] {caption}"
                elif self.args.stage == 2 and key == "references":
                    continue
                    filenames = []
                    for j, ref_artifact in enumerate(artifact):
                        filenames.append("references-"+f"{self.args.stage}-{j}-{ORIGINAL_SKETCH_PATH}.{ext}")
                elif key == "sketch_control":
                    continue
                    filename = f"sketch_control-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.{ext}"
                    caption = f"[sketch_control] {caption}"


                print("Filename after: ", filename)
                output_filename = os.path.join(self.args.output_dir, filename)
                
                print("Output filename: ", output_filename)

                if parallel_backend.is_main_process and ext in ["mp4", "jpg", "jpeg", "png"]:
                    main_process_prompts_to_filenames[PROMPT] = filename

                if isinstance(artifact, data.ImageArtifact):
                    artifact.value.save(output_filename)
                    all_processes_artifacts.append(wandb.Image(output_filename, caption=PROMPT))
                elif isinstance(artifact, data.VideoArtifact):
                    export_to_video(artifact.value, output_filename, fps=EXPORT_FPS)
                    all_processes_artifacts.append(wandb.Video(output_filename, caption=PROMPT))
                elif isinstance(artifact, list):
                    for j, ref_artifact in enumerate(artifact):
                        ref_output_filename = os.path.join(self.args.output_dir, f"references-{self.args.stage}-{j}-{ORIGINAL_SKETCH_PATH}.{ext}")
                        ref_artifact.value.save(ref_output_filename)
                        #all_processes_artifacts.append(wandb.Image(ref_output_filename, caption=PROMPT))

        # 3. Cleanup & log artifacts
        parallel_backend.wait_for_everyone()

        memory_statistics = utils.get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")

        # Remove all hooks that might have been added during pipeline initialization to the models
        pipeline.remove_all_hooks()
        del pipeline
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "image_processor", "vae"]
        if self.args.enable_precomputation:
            self._delete_components(module_names)
        torch.cuda.reset_peak_memory_stats(parallel_backend.device)

        # Gather artifacts from all processes. We also need to flatten them since each process returns a list of artifacts.
        # TODO(aryan): probably should only all gather from dp mesh process group
        all_artifacts = [None] * parallel_backend.world_size
        if parallel_backend.world_size > 1:
            torch.distributed.all_gather_object(all_artifacts, all_processes_artifacts)
        else:
            # TODO(aryan): workaround for accelerate for now, but refactor
            all_artifacts = [all_processes_artifacts]
        all_artifacts = [artifact for artifacts in all_artifacts for artifact in artifacts]

        if parallel_backend.is_main_process:
            tracker_key = "final" if final_validation else "validation"
            artifact_log_dict = {}

            image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
            if len(image_artifacts) > 0:
                artifact_log_dict["images"] = image_artifacts
            video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
            if len(video_artifacts) > 0:
                artifact_log_dict["videos"] = video_artifacts

            #TODO(Bryan): Turn of wandb artifact for now
            #parallel_backend.log({tracker_key: artifact_log_dict}, step=step)

            if self.args.push_to_hub and final_validation:
                video_filenames = list(main_process_prompts_to_filenames.values())
                prompts = list(main_process_prompts_to_filenames.keys())
                utils.save_model_card(
                    args=self.args, repo_id=self.state.repo_id, videos=video_filenames, validation_prompts=prompts
                )

        parallel_backend.wait_for_everyone()
        if not final_validation:
            self._move_components_to_device()
            self.transformer.train()

    def _eval(self)->None:
        if not self.args.disable_validation:
            #TODO (Bryan): Fix the bug with Final Validation: OOM
            #### TODO (Bryan): Erase hardcode of the self.state.train_state.step
            logger.info("Type of args.resume from checkpoint: ")
            logger.info(self.args.resume_from_checkpoint)
            logger.info(type(self.args.resume_from_checkpoint))
            logger.info("Type of train state: ")
            logger.info(self.state.train_state.step)
            logger.info(type(self.state.train_state.step))
            self.state.train_state.step=int(self.args.resume_from_checkpoint)
            self._validate(step=self.args.resume_from_checkpoint, final_validation=True, run_from_eval=True)

    def _init_distributed(self) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        print("world size is: ", world_size)

        # TODO(aryan): handle other backends
        backend_cls: parallel.ParallelBackendType = parallel.get_parallel_backend_cls(self.args.parallel_backend)
        self.state.parallel_backend = backend_cls(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
            backend="nccl",
            timeout=self.args.init_timeout,
            logging_dir=self.args.logging_dir,
            output_dir=self.args.output_dir,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        if self.args.seed is not None:
            self.state.parallel_backend.enable_determinism(self.args.seed)

    def _init_logging(self) -> None:
        logging._set_parallel_backend(self.state.parallel_backend)
        logging.set_dependency_log_level(self.args.verbose, self.state.parallel_backend.is_local_main_process)
        logger.info("Initialized FineTrainers")

    def _init_trackers(self) -> None:
        # TODO(aryan): handle multiple trackers
        trackers = [self.args.report_to]
        experiment_name = self.args.tracker_name or "finetrainers-experiment"
        self.state.parallel_backend.initialize_trackers(
            trackers, experiment_name=experiment_name, config=self._get_training_info(), log_dir=self.args.logging_dir
        )

    def _init_directories_and_repositories(self) -> None:
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)

            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, repo_id=repo_id, exist_ok=True).repo_id

    def _init_config_options(self) -> None:
        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision(self.args.float32_matmul_precision)

    def _move_components_to_device(
        self, components: Optional[List[torch.nn.Module]] = None, device: Optional[Union[str, torch.device]] = None
    ) -> None:
        if device is None:
            device = self.state.parallel_backend.device
        if components is None:
            components = [
                self.text_encoder,
                self.text_encoder_2,
                self.text_encoder_3,
                self.image_encoder,
                self.transformer,
                self.vae,
            ]
        components = utils.get_non_null_items(components)
        components = list(filter(lambda x: hasattr(x, "to"), components))
        for component in components:
            component.to(device)

    def _set_components(self, components: Dict[str, Any]) -> None:
        for component_name in self._all_component_names:
            existing_component = getattr(self, component_name, None)
            new_component = components.get(component_name, existing_component)
            setattr(self, component_name, new_component)

    def _delete_components(self, component_names: Optional[List[str]] = None) -> None:
        if component_names is None:
            component_names = self._all_component_names
        for component_name in component_names:
            setattr(self, component_name, None)
        utils.free_memory()
        utils.synchronize_device()

    def _init_pipeline(self, final_validation: bool = False) -> DiffusionPipeline:
        module_names = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "transformer", "vae"]

        if not final_validation:
            module_names.remove("transformer")
            pipeline = self.model_specification.load_pipeline(
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                tokenizer_3=self.tokenizer_3,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                text_encoder_3=self.text_encoder_3,
                image_encoder=self.image_encoder,
                image_processor=self.image_processor,
                # TODO(aryan): handle unwrapping for compiled modules
                # transformer=utils.unwrap_model(accelerator, self.transformer),
                transformer=self.transformer,
                vae=self.vae,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=True,
            )
        else:
            self._delete_components()

            # Load the transformer weights from the final checkpoint if performing full-finetune
            transformer = None
            if self.args.training_type == TrainingType.FULL_FINETUNE or self.args.training_type == TrainingType.TEMPORAL_FULL_FINETUNE:
                #TODO(Bryan): Implement better loading method, do not force load here
                checkpoint_path = str(os.path.join(self.args.output_dir, "model_weights", f"{self.state.train_state.step:06d}"))
                transformer = self.model_specification.load_diffusion_models(checkpoint_path)["transformer"]

            pipeline = self.model_specification.load_pipeline(
                transformer=transformer,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=False,
            )

            # Load the LoRA weights if performing LoRA finetuning
            if self.args.training_type == TrainingType.LORA or self.args.training_type == TrainingType.TEMPORAL_LORA:
                pipeline.load_lora_weights(
                    os.path.join(self.args.output_dir, "lora_weights", f"{self.state.train_state.step:06d}")
                )

        components = {module_name: getattr(pipeline, module_name, None) for module_name in module_names}
        self._set_components(components)
        if not self.args.enable_model_cpu_offload:
            self._move_components_to_device(list(components.values()))
        self._maybe_torch_compile()
        return pipeline

    def _prepare_data(
        self,
        preprocessor: Union[data.InMemoryDistributedDataPreprocessor, data.PrecomputedDistributedDataPreprocessor],
        data_iterator,
    ):
        if not self.args.enable_precomputation:
            if not self._are_condition_models_loaded:
                logger.info(
                    "Precomputation disabled. Loading in-memory data loaders. All components will be loaded on GPUs."
                )
                condition_components = self.model_specification.load_condition_models()
                latent_components = self.model_specification.load_latent_models()
                all_components = {**condition_components, **latent_components}
                self._set_components(all_components)
                self._move_components_to_device(list(all_components.values()))
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                self._maybe_torch_compile()
            else:
                condition_components = {k: v for k in self._condition_component_names if (v := getattr(self, k, None))}
                latent_components = {k: v for k in self._latent_component_names if (v := getattr(self, k, None))}

            condition_iterator = preprocessor.consume(
                "condition",
                components=condition_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                cache_samples=True,
            )
            latent_iterator = preprocessor.consume(
                "latent",
                components=latent_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                use_cached_samples=True,
                drop_samples=True,
            )

            self._are_condition_models_loaded = True
        else:
            #enter here
            logger.info("Precomputed condition & latent data exhausted. Loading & preprocessing new data.")

            parallel_backend = self.state.parallel_backend
            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer], "cpu")
                utils.free_memory()
                utils.synchronize_device()
                torch.cuda.reset_peak_memory_stats(parallel_backend.device)

            if self.args.precomputation_once:
                #(Bryan) erase this hardcoded later
                raise ValueError("Should not enter here")
                consume_fn = preprocessor.consume_once
            else:
                consume_fn = preprocessor.consume

            # Prepare condition iterators
            condition_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                #(Bryan): CogVideoX will not enter here
                condition_components = self.model_specification.load_condition_models()
                component_names = list(condition_components.keys())
                component_modules = list(condition_components.values())
                self._set_components(condition_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            condition_iterator = consume_fn(
                "condition",
                components=condition_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                cache_samples=True,
            )
            self._delete_components(component_names)
            del condition_components, component_names, component_modules

            # Prepare latent iterators
            latent_components, component_names, component_modules = {}, [], []
            if not self.args.precomputation_reuse:
                #(Bryan): CogVideoX will not enter here
                latent_components = self.model_specification.load_latent_models()
                utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
                component_names = list(latent_components.keys())
                component_modules = list(latent_components.values())
                self._set_components(latent_components)
                self._move_components_to_device(component_modules)
                self._maybe_torch_compile()
            latent_iterator = consume_fn(
                "latent",
                components=latent_components,
                data_iterator=data_iterator,
                generator=self.state.generator,
                use_cached_samples=True,
                drop_samples=True,
            )
            self._delete_components(component_names)
            del latent_components, component_names, component_modules

            if parallel_backend.world_size == 1:
                self._move_components_to_device([self.transformer])

        return condition_iterator, latent_iterator

    def _maybe_torch_compile(self):
        for model_name, compile_scope in zip(self.args.compile_modules, self.args.compile_scopes):
            model = getattr(self, model_name, None)
            if model is not None:
                logger.info(f"Applying torch.compile to '{model_name}' with scope '{compile_scope}'.")
                compiled_model = utils.apply_compile(model, compile_scope)
                setattr(self, model_name, compiled_model)

    def _get_training_info(self) -> Dict[str, Any]:
        info = self.args.to_dict()

        # Removing flow matching arguments when not using flow-matching objective
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info

    # fmt: off
    _all_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder", "image_processor", "transformer", "unet", "vae", "scheduler"]
    _condition_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3"]
    _latent_component_names = ["image_encoder", "image_processor", "vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]#
    # fmt: on
