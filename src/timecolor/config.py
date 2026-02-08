from enum import Enum
from typing import Type

from .models import ModelSpecification
from .models.cogvideox import TimecolorModelSpecification


class ModelType(str, Enum):
    COGVIDEOX = "cogvideox"
    COGVIEW4 = "cogview4"
    FLUX = "flux"
    HUNYUAN_VIDEO = "hunyuan_video"
    LTX_VIDEO = "ltx_video"
    WAN = "wan"


class TrainingType(str, Enum):
    # SFT
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"

    # Control
    CONTROL_LORA = "control-lora"
    CONTROL_FULL_FINETUNE = "control-full-finetune"

    #temporal concat
    TEMPORAL_LORA = "temporal-lora"
    TEMPORAL_FULL_FINETUNE = "temporal-full-finetune"


SUPPORTED_MODEL_CONFIGS = {
    # TODO(aryan): autogenerate this
    # SFT
    ModelType.COGVIDEOX: {
        TrainingType.TEMPORAL_FULL_FINETUNE: TimecolorModelSpecification
    },
}


def _get_model_specifiction_cls(model_name: str, training_type: str) -> Type[ModelSpecification]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
