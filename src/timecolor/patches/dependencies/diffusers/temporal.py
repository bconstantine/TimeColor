from contextlib import contextmanager
from typing import List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook


_TEMPORAL_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_TEMPORAL_CHANNEL_CONCATENATE_HOOK"
_TEMPORAL_CHANNEL_TIMEEMBEDDING_HOOK = "FINETRAINERS_TEMPORAL_CHANNEL_TIMEEMBEDDING_HOOK"
_CROP_LATENTS_HOOK = "FINETRAINERS_CROP_LATENTS_HOOK"


class TemporalChannelConcatenateHook(ModelHook):
    def __init__(self, input_names: List[str], inputs: List[torch.Tensor], dims: List[int]):
        self.input_names = input_names
        self.inputs = inputs
        self.dims = dims

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        for input_name, input_tensor, dim in zip(self.input_names, self.inputs, self.dims):
            original_tensor = args[input_name] if isinstance(input_name, int) else kwargs[input_name]
            control_tensor = torch.cat([original_tensor, input_tensor], dim=dim)
            if isinstance(input_name, int):
                args[input_name] = control_tensor
            else:
                kwargs[input_name] = control_tensor
        return args, kwargs


@contextmanager
def temporal_channel_concat(
    module: torch.nn.Module, input_names: List[Union[int, str]], inputs: List[torch.Tensor], dims: List[int]
):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = TemporalChannelConcatenateHook(input_names, inputs, dims)
    registry.register_hook(hook, _TEMPORAL_CHANNEL_CONCATENATE_HOOK)
    yield
    registry.remove_hook(_TEMPORAL_CHANNEL_CONCATENATE_HOOK, recurse=False)


class KwargOverrideHook(ModelHook):
    def __init__(self, overrides: dict):
        self.overrides = overrides

    def pre_forward(self, module, *args, **kwargs):
        kwargs.update(self.overrides)
        return args, kwargs

@contextmanager
def override_kwargs(module, **overrides):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = KwargOverrideHook(overrides)
    registry.register_hook(hook, _TEMPORAL_CHANNEL_TIMEEMBEDDING_HOOK)
    yield
    registry.remove_hook(_TEMPORAL_CHANNEL_TIMEEMBEDDING_HOOK, recurse=False)

class CropFramesHook(ModelHook):
    def __init__(self, out_index: int = 0, keep: int = 5, dim: int = 1):
        """
        out_index : which element of the tuple/list to crop
        keep      : number of frames to keep
        dim       : temporal dimension (default 1 for (B,F,C,H,W))
        """
        self.out_index = out_index
        self.keep = keep
        self.dim = dim

    def post_forward(self, module, output, *args, **kwargs):
        # output can be tensor or tuple/list
        if isinstance(output, (tuple, list)):
            out = list(output)
            out[self.out_index] = out[self.out_index].narrow(self.dim, 0, self.keep)
            return tuple(out)
        else:  # single tensor
            return output.narrow(self.dim, 0, self.keep)

@contextmanager
def crop_output_frames(module: torch.nn.Module, keep: int = 5, out_index: int = 0, dim: int = 1):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = CropFramesHook(out_index=out_index, keep=keep, dim=dim)
    registry.register_hook(hook, _CROP_LATENTS_HOOK)
    yield
    registry.remove_hook(_CROP_LATENTS_HOOK, recurse=False)