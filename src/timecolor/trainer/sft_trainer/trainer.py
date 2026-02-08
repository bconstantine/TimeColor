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

    def eval(self) -> None:
        try:
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

    def _validate(self, step: int, final_validation: bool = False, run_from_eval: bool = False) -> None:
        if self.args.validation_dataset_file is None:
            logger.info("Warning, dataset is empty! Cancelling inference")
            return

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
            PROMPT = validation_data["prompt"]
            caption = PROMPT
            if self.args.stage == 0:
                STARTING_FULL_FRAME_REF_IMAGE = validation_data["starting_full_frame_ref"]
            elif self.args.stage == 1:
                FULL_FRAME_REF_IMAGE = validation_data["full_frame_ref"]
            elif self.args.stage == 2:
                REFERENCES = validation_data["references"]
            else:
                raise ValueError(f"Unsupported stage {self.args.stage} for validation dataset.")

            
            #file naming for video output
            filename = ""
            if CUSTOM_FILE_NAME:
                filename += CUSTOM_FILE_NAME
            else:
                filename += f"out-{self.args.stage}-{ORIGINAL_SKETCH_PATH}.mp4"
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
                artifacts = {
                    "references": [data.ImageArtifact(value=ref) for ref in REFERENCES],
                    "sketch_control": data.VideoArtifact(value=SKETCH_CONTROL_VIDEO),
                }


            # 2.2. Track the artifacts generated from validation
            for i, validation_artifact in enumerate(validation_artifacts):
                if validation_artifact.value is None:
                    continue
                artifacts.update({f"artifact_{i}": validation_artifact})
            for index, (key, artifact) in enumerate(list(artifacts.items())):
                assert isinstance(artifact, (data.ImageArtifact, data.VideoArtifact, list))
                if isinstance(artifact, list):
                    assert all(isinstance(a, data.ImageArtifact) for a in artifact), "All artifacts in the list must be ImageArtifact"
                    ext = artifact[0].file_extension
                elif artifact.value is None:
                    continue
                else:
                    try:
                        ext = artifact.file_extension
                    except AttributeError:
                        print("Artifact value does not have file_extension, the value of artifact value is: ", artifact.value)
                        raise ValueError("Artifact value must have file_extension attribute")

                time_, rank = int(time.time()), parallel_backend.rank
                
                print("Filename before: ", filename)

                if key in ["starting_full_frame_ref", "full_frame_ref", "identity_mask", "references", "sketch_control"]:
                    continue #avoid saving input for release code


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

        parallel_backend.wait_for_everyone()
        if not final_validation:
            self._move_components_to_device()
            self.transformer.train()

    def _eval(self)->None:
        self._validate(step=None, final_validation=True, run_from_eval=True)

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
            assert NotImplementedError("This should only be entered when validation in training phase")
        else:
            self._delete_components()

            # Load the transformer weights from the final checkpoint if performing full-finetune
            transformer = None
            checkpoint_path = str(os.path.join(self.args.output_dir, "model_weights"))
            transformer = self.model_specification.load_diffusion_models(checkpoint_path)["transformer"]

            pipeline = self.model_specification.load_pipeline(
                transformer=transformer,
                enable_slicing=self.args.enable_slicing,
                enable_tiling=self.args.enable_tiling,
                enable_model_cpu_offload=self.args.enable_model_cpu_offload,
                training=False,
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
