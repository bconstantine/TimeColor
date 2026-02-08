import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Iterable
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from .temporal_pipeline import CogVideoXTemporalPipeline
from PIL.Image import BICUBIC
from PIL.Image import Image
from PIL import ImageDraw, ImageOps
from PIL.Image import new as ImageNew
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import numpy as np
import math
import gc

from timecolor.data import VideoArtifact
from timecolor.logging import get_logger
from timecolor.models.modeling_utils import ModelSpecification
from timecolor.models.utils import DiagonalGaussianDistribution
from timecolor.processors import ProcessorMixin, T5Processor
from timecolor.typing import ArtifactType, SchedulerType
from timecolor.utils import _enable_vae_memory_optimizations, get_non_null_items
from timecolor.patches.dependencies.diffusers.temporal import temporal_channel_concat, \
    override_kwargs, crop_output_frames
from .utils import prepare_rotary_positional_embeddings_cogvideox, concatenate_freqs_cos_freqs_sin_temporally


logger = get_logger()


""""
PARSING CONSTANTS FOR THE COLORIZATION
"""
SPACE_STRIDE = 8   # H/8, W/8  (latent grid)
TIME_STRIDE = 4    # after first frame: [1..4],[5..8],...
PATCH_SIZE = 2     # latent -> attention grid (downsample by 2)
# imageio returns RGB frames
# IDs: 0 black, 1 blue, 2 green, 3 red
PALETTE_RGB = np.array([
    [0,   0,   0  ],
    [0,   0,   255],
    [0,   255, 0  ],
    [255, 0,   0  ],
], dtype=np.int16)
def build_time_groups(T: int, time_stride: int = TIME_STRIDE) -> List[List[int]]:
    if T <= 0:
        return []
    groups: List[List[int]] = []
    groups.append([0])
    t = 1
    while t < T:
        end = min(T, t + time_stride)
        groups.append(list(range(t, end)))
        t = end
    return groups

def majority_vote_downsample_spatial(mask2d: np.ndarray, out_h: int, out_w: int, block: int) -> np.ndarray:
    H, W = mask2d.shape
    H_crop = out_h * block
    W_crop = out_w * block
    if H_crop != H or W_crop != W:
        mask2d = mask2d[:H_crop, :W_crop]

    K = int(mask2d.max())
    counts = np.zeros((K + 1, out_h, out_w), dtype=np.int32)
    for label in range(K + 1):
        m = (mask2d == label).astype(np.int32)
        m = m.reshape(out_h, block, out_w, block)
        counts[label] = m.sum(axis=(1, 3))
    return counts.argmax(axis=0).astype(np.uint8)

def map_real_to_latent_index(t_real: int, time_stride: int, t_lat_max: int) -> int:
    if t_real == 0:
        idx = 0
    else:
        idx = 1 + (t_real - 1) // time_stride
    return max(0, min(idx, t_lat_max - 1))
def downsample_to_latent(mask_17: np.ndarray, time_stride: int, space_stride: int) -> np.ndarray:
    assert mask_17.ndim == 3
    T, H, W = mask_17.shape

    time_groups = build_time_groups(T, time_stride=time_stride)
    T_lat = len(time_groups)

    H_lat = H // space_stride
    W_lat = W // space_stride
    H_crop = H_lat * space_stride
    W_crop = W_lat * space_stride
    if H_crop != H or W_crop != W:
        mask_17 = mask_17[:, :H_crop, :W_crop]

    K = int(mask_17.max())
    latent_mask = np.zeros((T_lat, H_lat, W_lat), dtype=np.uint8)

    for t_lat, frames in enumerate(time_groups):
        group = mask_17[frames]  # (F, H_crop, W_crop)
        F = group.shape[0]
        counts = np.zeros((K + 1, H_lat, W_lat), dtype=np.int32)

        for label in range(K + 1):
            vol = (group == label).astype(np.int32)
            vol = vol.reshape(F, H_lat, space_stride, W_lat, space_stride)
            counts[label] = vol.sum(axis=(0, 2, 4))

        latent_mask[t_lat] = counts.argmax(axis=0).astype(np.uint8)

    return latent_mask

def downsample_latent_to_attention(latent_mask: np.ndarray, patch_size: int) -> np.ndarray:
    assert latent_mask.ndim == 3
    T_lat, H_lat, W_lat = latent_mask.shape

    H_att = H_lat // patch_size
    W_att = W_lat // patch_size
    H_crop = H_att * patch_size
    W_crop = W_att * patch_size
    if H_crop != H_lat or W_crop != W_lat:
        latent_mask = latent_mask[:, :H_crop, :W_crop]

    attn_mask = np.zeros((T_lat, H_att, W_att), dtype=np.uint8)
    for t in range(T_lat):
        attn_mask[t] = majority_vote_downsample_spatial(
            latent_mask[t], out_h=H_att, out_w=W_att, block=patch_size
        )
    return attn_mask

def ids_from_palette_tol(frame_rgb_u8: np.ndarray, tol: int) -> np.ndarray:
    """
    (H,W,3) uint8 RGB -> (H,W) uint8 in {0..3}
    """
    f = frame_rgb_u8.astype(np.int16, copy=False)
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) RGB frame, got {f.shape}")

    H, W, _ = f.shape
    ids = np.full((H, W), 255, dtype=np.uint8)

    for idx, (r, g, b) in enumerate(PALETTE_RGB):
        diff = np.abs(f - np.array([r, g, b], dtype=np.int16))
        m = (diff[..., 0] <= tol) & (diff[..., 1] <= tol) & (diff[..., 2] <= tol)
        ids[m] = idx

    unk = (ids == 255)
    if np.any(unk):
        d2 = ((f[..., None, :] - PALETTE_RGB[None, None, :, :]) ** 2).sum(axis=-1)  # (H,W,4)
        ids[unk] = d2[unk].argmin(axis=-1).astype(np.uint8)

    return ids

def ids_to_rgb(ids_2d: np.ndarray) -> np.ndarray:
    return PALETTE_RGB[ids_2d].astype(np.uint8)  # (H,W,3) RGB

VideoLike = Union[
    Image,                 # single frame
    Sequence[Image],       # multiple frames
    np.ndarray,                  # (T,H,W,3) or (H,W,3)
]

def make_attention_id_mask_from_mask_mp4(
    final_17correspondence_video: Optional[VideoLike] = None,
    *,
    tol: int = 13,
    time_stride: int = TIME_STRIDE,
    space_stride: int = SPACE_STRIDE,
    patch_size: int = PATCH_SIZE,
    keep_last: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Input:
      final_17correspondence_video:
        - Sequence[PIL.Image.Image] with length T (e.g., 17), each RGB
        - OR numpy array (T,H,W,3) RGB uint8
        - OR single PIL Image / single (H,W,3) array (treated as T=1)

    Returns:
      real_ids   : (T,H,W) uint8
      latent_ids : (T_lat,H_lat,W_lat) uint8
      attn_ids   : (T_lat,H_att,W_att) uint8
    """
    if final_17correspondence_video is None:
        raise ValueError("final_17correspondence_video is None.")

    # -------- normalize input to list of RGB uint8 frames --------
    frames_rgb: list[np.ndarray] = []

    if isinstance(final_17correspondence_video, Image):
        # single frame
        frame = final_17correspondence_video.convert("RGB")
        frames_rgb = [np.asarray(frame, dtype=np.uint8)]
    elif isinstance(final_17correspondence_video, np.ndarray):
        arr = final_17correspondence_video
        if arr.ndim == 3:
            # (H,W,3)
            if arr.shape[2] != 3:
                raise ValueError(f"Expected (H,W,3), got {arr.shape}")
            frames_rgb = [arr.astype(np.uint8, copy=False)]
        elif arr.ndim == 4:
            # (T,H,W,3)
            if arr.shape[-1] != 3:
                raise ValueError(f"Expected (T,H,W,3), got {arr.shape}")
            arr_u8 = arr.astype(np.uint8, copy=False)
            frames_rgb = [arr_u8[t] for t in range(arr_u8.shape[0])]
        else:
            raise ValueError(f"Unexpected ndarray shape: {arr.shape}")
    else:
        # sequence of PIL Images
        seq = final_17correspondence_video  # type: ignore[assignment]
        for im in seq:  # type: ignore[arg-type]
            if not isinstance(im, Image):
                raise ValueError(f"Expected PIL.Image in sequence, got {type(im)}")
            frames_rgb.append(np.asarray(im.convert("RGB"), dtype=np.uint8))

    if len(frames_rgb) == 0:
        raise RuntimeError("No frames provided.")

    # Optional: keep only last 17 frames, matching your validation "final_17..."
    if keep_last is not None and len(frames_rgb) > keep_last:
        frames_rgb = frames_rgb[-keep_last:]

    # Sanity: all frames same H,W
    H0, W0 = frames_rgb[0].shape[:2]
    for i, fr in enumerate(frames_rgb):
        if fr.shape[:2] != (H0, W0) or fr.ndim != 3 or fr.shape[2] != 3:
            raise ValueError(f"Frame {i} has shape {fr.shape}, expected {(H0, W0, 3)}")

    # -------- per-frame palette snapping -> real_ids (T,H,W) --------
    ids_list = [ids_from_palette_tol(fr, tol=tol) for fr in frames_rgb]  # each (H,W)
    real_ids = np.stack(ids_list, axis=0).astype(np.uint8)  # (T,H,W)

    # -------- downsample to latent & attention --------
    latent_ids = downsample_to_latent(real_ids, time_stride=time_stride, space_stride=space_stride)
    attn_ids = downsample_latent_to_attention(latent_ids, patch_size=patch_size)
    return real_ids, latent_ids, attn_ids

def make_attention_id_mask_from_mask_npz(
    mask_npz: np.ndarray,
    *,
    time_stride: int = TIME_STRIDE,
    space_stride: int = SPACE_STRIDE,
    patch_size: int = PATCH_SIZE,
    keep_last: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same downsample pipeline as make_attention_id_mask_from_mask_mp4(),
    but input is already an ID mask in (T,H,W) (e.g., output of _load_npz_mask).

    Returns:
      real_ids   : (T,H,W) uint8
      latent_ids : (T_lat,H_lat,W_lat) uint8
      attn_ids   : (T_lat,H_att,W_att) uint8
    """
    real_ids = np.asarray(mask_npz)
    if real_ids.ndim != 3:
        raise ValueError(f"Expected (T,H,W) mask, got {real_ids.shape}")

    # Optional: keep only last frames
    if keep_last is not None and keep_last >= 0 and real_ids.shape[0] > keep_last:
        real_ids = real_ids[-keep_last:]

    # Ensure integer labels and compact dtype
    if real_ids.dtype.kind not in ("i", "u"):
        real_ids = real_ids.astype(np.int64, copy=False)

    maxv = int(real_ids.max()) if real_ids.size else 0
    if maxv > 255:
        raise ValueError(f"Mask values too large for uint8: max={maxv}")

    real_ids = real_ids.astype(np.uint8, copy=False)

    # Same operations as before
    latent_ids = downsample_to_latent(real_ids, time_stride=time_stride, space_stride=space_stride)
    attn_ids = downsample_latent_to_attention(latent_ids, patch_size=patch_size)
    return real_ids, latent_ids, attn_ids


def _flatten_thw_indices(t: np.ndarray, h: np.ndarray, w: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
    """
    Flatten (t,h,w) -> flat index with CogVideoX order:
      flat = t*(Hp*Wp) + h*Wp + w
    """
    return t * (Hp * Wp) + h * Wp + w


def _load_npz_mask(mask_path: Path, key: str = "mask") -> np.ndarray:
    npz = np.load(mask_path)
    if key not in npz:
        raise RuntimeError(f"'{key}' key not found in {mask_path}")
    arr = npz[key]
    if not isinstance(arr, np.ndarray):
        raise RuntimeError(f"{mask_path}: '{key}' is not a numpy array")
    if arr.ndim != 3:
        raise RuntimeError(f"{mask_path}: expected 3D (T,H,W), got {arr.shape}")
    return arr

    
def make_black_pil_video(frames: int,
    width: int,
    height: int,
    rgb: int = 0) -> List[Image]:
    rgb = int(np.clip(rgb, 0, 255))
    return [ImageNew("RGB", (width, height), (rgb, rgb, rgb)) for _ in range(frames)]

class CogVideoXLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the CogVideoX VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 1

    def forward(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            if vae.use_slicing and video.shape[0] > 1:
                encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
                moments = torch.cat(encoded_slices)
            else:
                moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W] -> [B, F, C, H, W]
        return {self.output_names[0]: latents}

class TimecolorModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "prompt_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [CogVideoXLatentEncodeProcessor(["latents"])]

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

        self.attention_mask_stage = 3

    @property
    def _resolution_dim_keys(self):
        return {"latents": (1, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLCogVideoX.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self, custom_path: str = None) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            loaded_path = custom_path if custom_path else self.pretrained_model_name_or_path
            logger.info("Loading from: ")
            logger.info(loaded_path)
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                loaded_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", **common_kwargs
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        transformer: Optional[CogVideoXTransformer3DModel] = None,
        vae: Optional[AutoencoderKLCogVideoX] = None,
        scheduler: Optional[CogVideoXDDIMScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> CogVideoXPipeline:
        print("Entering load pipeline")
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)
        
        print("Enable slicig and tiling is: ", enable_slicing, enable_tiling)
        print("Enable model cpu offload is: ", enable_model_cpu_offload)

        # pipe = CogVideoXPipeline.from_pretrained(
        #     self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        # )

        pipe = CogVideoXTemporalPipeline.from_pretrained(
            self.pretrained_model_name_or_path, **components, revision=self.revision, cache_dir=self.cache_dir
        )
        print("Initialized cogvideox temporal pipeline")
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 226,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        conditions.pop("prompt_attention_mask", None)
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image": image,
            "video": video,
            "generator": generator,
            "compute_posterior": compute_posterior,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def validation(
        self,
        pipeline: CogVideoXPipeline,
        prompt: str,
        starting_full_frame_ref: Optional[Image] = None,
        full_frame_ref: Optional[Image] = None,
        sketch_control: Optional[Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_enumeration_method: str = "INVALID",
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        identity_mask_coloredvid: Optional[Image] = None,
        identity_mask_npz: Optional[np.ndarray] = None,
        generator: Optional[torch.Generator] = None,
        stage: int = 1,
        references: Optional[List[Image]] = None,
        guidance_scale = 3, 
        **kwargs,
    ) -> List[ArtifactType]:
        if stage == 0 or stage == 1:
            number_of_subject = 0
        elif stage == 2: 
            number_of_subject = len(references)-1
        else:
            raise ValueError(f"Stage {stage} not supported. Only 1 and 2 are supported.")
        orig_sketch_control = sketch_control
        if stage == 0:
            assert starting_full_frame_ref is not None, "starting_full_frame_ref must be provided for stage 0"
            assert sketch_control is not None, "sketch_control must be provided for stage 0"

            with torch.no_grad():
                dtype = pipeline.vae.dtype
                device = pipeline._execution_device

                starting_full_frame_ref = pipeline.video_processor.preprocess(starting_full_frame_ref, height=height, width=width)
                starting_full_frame_ref = starting_full_frame_ref.to(device=device, dtype=dtype)
                sketch_control = pipeline.video_processor.preprocess_video(sketch_control, height=height, width=width)
                sketch_control = sketch_control.to(device=device, dtype=dtype)


                starting_full_frame_ref = starting_full_frame_ref.unsqueeze(2) # B C 1 H W
                starting_full_frame_ref = pipeline.vae.encode(starting_full_frame_ref).latent_dist.sample(generator=generator)
                sketch_control = pipeline.vae.encode(sketch_control).latent_dist.sample(generator=generator)

            B, C, F, H, W = sketch_control.shape
            
            starting_full_frame_ref = self.vae_config.scaling_factor * starting_full_frame_ref
            sketch_control = self.vae_config.scaling_factor * sketch_control
            
            patch_size = self.transformer_config.patch_size
            patch_size_t = getattr(self.transformer_config, "patch_size_t", None)
            VAE_SPATIAL_SCALE_FACTOR = 8
            rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
            rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
            final_frame_length = 2*F+1+number_of_subject
            OFFSET_W_MULTIPLIER = (W * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            OFFSET_H_MULTIPLIER = (H * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            
            image_rotary_emb = self.implement_rope_custom(rope_enumeration_method=rope_enumeration_method,
                              vidpart_latent_height=H,
                              vidpart_latent_width=W,
                              vidpart_latent_frame_length=F,
                              final_frame_length=final_frame_length,
                              vae_spatial_scale_factor=VAE_SPATIAL_SCALE_FACTOR,
                              patch_size=patch_size,
                              patch_size_t=patch_size_t,
                              device=device,
                              rope_base_height=rope_base_height,
                              rope_base_width=rope_base_width,
                              offset_w_multiplier=OFFSET_W_MULTIPLIER,
                              offset_h_multiplier=OFFSET_H_MULTIPLIER,
                              number_of_subject=number_of_subject)

            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "return_dict": True,
                "output_type": "pil",
                "guidance_scale": guidance_scale,
            }
            generation_kwargs = get_non_null_items(generation_kwargs)
            token_spans = self._recover_token_spans(
                stage=stage,
                F=F,
                H=H,
                W=W,
                patch_size=patch_size,
                number_of_subject=number_of_subject,
                device=device,
            )

            # starting_full_frame_ref = starting_full_frame_ref.permute(0, 2, 1, 3, 4)
            # sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            # if guidance_scale > 1:
            #     starting_full_frame_ref = torch.concat([starting_full_frame_ref]*2, dim=0)
            #     sketch_control = torch.concat([sketch_control]*2, dim=0)
            # )
            # with temporal_channel_concat(pipeline.transformer, 
            #                              ["hidden_states", "hidden_states"], 
            #                              [sketch_control, starting_full_frame_ref], 
            #                              dims=[1,1,1]), \
            # override_kwargs(pipeline.transformer, image_rotary_emb = image_rotary_emb, attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}), \
            #     crop_output_frames(pipeline.transformer, keep=F, out_index=0, dim=1):
            #     video = pipeline(**generation_kwargs).frames[0]

            #try using temporal pipelines
            starting_full_frame_ref = starting_full_frame_ref.permute(0, 2, 1, 3, 4)
            sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            video = pipeline(**generation_kwargs, 
                                sketch_control_hiddenstates = sketch_control, 
                                subjects_refs_hiddenstates = starting_full_frame_ref, 
                                image_rotary_emb = image_rotary_emb, 
                                attention_kwargs = {"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}, 
                                keep_frames = F, 
                                out_index = 0,
                                out_dim = 1).frames[0]

            return [VideoArtifact(value=video)]
        elif stage == 1:
            assert full_frame_ref is not None, "full_frame_ref must be provided for stage 1"
            assert sketch_control is not None, "sketch_control must be provided for stage 1"

            with torch.no_grad():
                dtype = pipeline.vae.dtype
                device = pipeline._execution_device
                full_frame_ref = pipeline.video_processor.preprocess(full_frame_ref, height=height, width=width)
                full_frame_ref = full_frame_ref.to(device=device, dtype=dtype)
                sketch_control = pipeline.video_processor.preprocess_video(sketch_control, height=height, width=width)
                sketch_control = sketch_control.to(device=device, dtype=dtype)

                #Change full_frame_ref from B C H W to B C F H W
                full_frame_ref = full_frame_ref.unsqueeze(2) # B C 1 H W
                full_frame_ref = pipeline.vae.encode(full_frame_ref).latent_dist.sample(generator=generator)
                sketch_control = pipeline.vae.encode(sketch_control).latent_dist.sample(generator=generator)


                full_frame_ref = self.vae_config.scaling_factor * full_frame_ref
                sketch_control = self.vae_config.scaling_factor * sketch_control

            B, C, F, H, W = sketch_control.shape
            patch_size = self.transformer_config.patch_size
            patch_size_t = getattr(self.transformer_config, "patch_size_t", None)
            VAE_SPATIAL_SCALE_FACTOR = 8
            rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
            rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
            final_frame_length = 2*F+1+number_of_subject
            OFFSET_W_MULTIPLIER = (W * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            OFFSET_H_MULTIPLIER = (H * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            
            image_rotary_emb = self.implement_rope_custom(rope_enumeration_method=rope_enumeration_method,
                              vidpart_latent_height=H,
                              vidpart_latent_width=W,
                              vidpart_latent_frame_length=F,
                              final_frame_length=final_frame_length,
                              vae_spatial_scale_factor=VAE_SPATIAL_SCALE_FACTOR,
                              patch_size=patch_size,
                              patch_size_t=patch_size_t,
                              device=device,
                              rope_base_height=rope_base_height,
                              rope_base_width=rope_base_width,
                              offset_w_multiplier=OFFSET_W_MULTIPLIER,
                              offset_h_multiplier=OFFSET_H_MULTIPLIER,
                              number_of_subject=number_of_subject)

            token_spans = self._recover_token_spans(
                stage=stage,
                F=F,
                H=H,
                W=W,
                patch_size=patch_size,
                number_of_subject=number_of_subject,
                device=device,
            )


            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "return_dict": True,
                "output_type": "pil",
                "guidance_scale": guidance_scale,
            }
            generation_kwargs = get_non_null_items(generation_kwargs)
            
            # if guidance_scale > 1:
            #     full_frame_ref = torch.concat([full_frame_ref]*2, dim=0)
            #     sketch_control = torch.concat([sketch_control]*2, dim=0)
            # #control latents are BCFHW, convert to BFCHW
            # full_frame_ref = full_frame_ref.permute(0, 2, 1, 3, 4)
            # sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            # with temporal_channel_concat(pipeline.transformer, 
            #                              ["hidden_states", "hidden_states"], 
            #                              [sketch_control, full_frame_ref], 
            #                              dims=[1,1,1]), \
            # override_kwargs(pipeline.transformer, image_rotary_emb = image_rotary_emb, attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}), \
            #     crop_output_frames(pipeline.transformer, keep=F, out_index=0, dim=1):
            #     video = pipeline(**generation_kwargs).frames[0]

            #try using temporal pipelines
            full_frame_ref = full_frame_ref.permute(0, 2, 1, 3, 4)
            sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            video = pipeline(**generation_kwargs, 
                                sketch_control_hiddenstates = sketch_control, 
                                subjects_refs_hiddenstates = full_frame_ref, 
                                image_rotary_emb = image_rotary_emb, 
                                attention_kwargs = {"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}, 
                                keep_frames = F, 
                                out_index = 0,
                                out_dim = 1).frames[0]

            return [VideoArtifact(value=video)]
        elif stage == 2:
            assert references is not None, "references must be provided for stage 2"
            assert sketch_control is not None, "sketch_control must be provided for stage 2"

            with torch.no_grad():
                dtype = pipeline.vae.dtype
                device = pipeline._execution_device
                sketch_control = pipeline.video_processor.preprocess_video(sketch_control, height=height, width=width)
                sketch_control = sketch_control.to(device=device, dtype=dtype)
                references = [pipeline.video_processor.preprocess_video(ref, height=height, width=width) for ref in references]
                references = [ref.to(device=device, dtype=dtype) for ref in references]

                for ref in references:
                    ref = ref.unsqueeze(2) # B C 1 H W
                
                sketch_control = pipeline.vae.encode(sketch_control).latent_dist.sample(generator=generator)
                for i, ref in enumerate(references):
                    references[i] = pipeline.vae.encode(ref).latent_dist.sample(generator=generator)

                sketch_control = self.vae_config.scaling_factor * sketch_control
                references = [self.vae_config.scaling_factor * ref for ref in references]
                

            B, C, F, H, W = sketch_control.shape
            patch_size = self.transformer_config.patch_size
            patch_size_t = getattr(self.transformer_config, "patch_size_t", None)
            VAE_SPATIAL_SCALE_FACTOR = 8
            rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
            rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
            final_frame_length = 2*F+1+number_of_subject
            OFFSET_W_MULTIPLIER = (W * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            OFFSET_H_MULTIPLIER = (H * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
            
            image_rotary_emb = self.implement_rope_custom(rope_enumeration_method=rope_enumeration_method,
                              vidpart_latent_height=H,
                              vidpart_latent_width=W,
                              vidpart_latent_frame_length=F,
                              final_frame_length=final_frame_length,
                              vae_spatial_scale_factor=VAE_SPATIAL_SCALE_FACTOR,
                              patch_size=patch_size,
                              patch_size_t=patch_size_t,
                              device=device,
                              rope_base_height=rope_base_height,
                              rope_base_width=rope_base_width,
                              offset_w_multiplier=OFFSET_W_MULTIPLIER,
                              offset_h_multiplier=OFFSET_H_MULTIPLIER,
                              number_of_subject=number_of_subject)
            token_spans = self._recover_token_spans(
                        stage=stage,
                        F=F,
                        H=H,
                        W=W,
                        patch_size=patch_size,
                        number_of_subject=number_of_subject,
                        device=device
                    )

            if self.mask_cache.level == 4:
                #unravel token spans
                if identity_mask_npz is not None:
                    real_ids, latent_ids, attn_ids = make_attention_id_mask_from_mask_npz(identity_mask_npz)
                else:
                    real_ids, latent_ids, attn_ids = make_attention_id_mask_from_mask_mp4(identity_mask_coloredvid)

                token_spans = self.expand_token_spans_with_attention_pixel_pack_ontherun(
                    token_spans=token_spans,
                    attn_ids=attn_ids,
                    F=F,
                    H=H,
                    W=W,
                    device=device,
                    allowed_ids=[i for i in range(len(references))],
                    strict_shape = True,
                    strict_values = True,
                    main_key = "MAIN",
                )
                
            generation_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "return_dict": True,
                "output_type": "pil"
            }
            generation_kwargs = get_non_null_items(generation_kwargs)

            subjects_refs = torch.concat(references, dim=2)  # B, C, N, H, W

            # if guidance_scale > 1:
            #     references = torch.concat([references]*2, dim=0)
            #     sketch_control = torch.concat([sketch_control]*2, dim=0)
            # subjects_refs = subjects_refs.permute(0, 2, 1, 3, 4)
            # sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            # with temporal_channel_concat(pipeline.transformer, 
            #                             ["hidden_states", "hidden_states"], 
            #                             [sketch_control, subjects_refs], 
            #                             dims=[1,1,1]), \
            # override_kwargs(pipeline.transformer, image_rotary_emb = image_rotary_emb, attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}), \
            #     crop_output_frames(pipeline.transformer, keep=F, out_index=0, dim=1):
            #     video = pipeline(**generation_kwargs).frames[0]

            #try using temporal pipelines
            subjects_refs = subjects_refs.permute(0, 2, 1, 3, 4)
            sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            video = pipeline(**generation_kwargs, 
                                sketch_control_hiddenstates = sketch_control, 
                                subjects_refs_hiddenstates = subjects_refs, 
                                image_rotary_emb = image_rotary_emb, 
                                attention_kwargs = {"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}, 
                                keep_frames = F, 
                                out_index = 0,
                                out_dim = 1).frames[0]
            

            #OLD INFER METHOD
            #{"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}

            # generation_kwargs = {
            #     "prompt": prompt,
            #     "height": height,
            #     "width": width,
            #     "num_frames": num_frames,
            #     "num_inference_steps": num_inference_steps,
            #     "generator": generator,
            #     "return_dict": True,
            #     "output_type": "pil",
            #     "guidance_scale":2.0,
            # }
            # generation_kwargs = get_non_null_items(generation_kwargs)

            
            # references = torch.concat(references, dim=2)  # B, C, N, H, W
            # if guidance_scale > 1:
            #     references = torch.concat([references]*2, dim=0)
            #     sketch_control = torch.concat([sketch_control]*2, dim=0)

            # references = references.permute(0, 2, 1, 3, 4)
            # sketch_control = sketch_control.permute(0, 2, 1, 3, 4)
            

            # with temporal_channel_concat(pipeline.transformer, 
            #                              ["hidden_states", "hidden_states"], 
            #                              [sketch_control, references], 
            #                              dims=[1,1,1]), \
            # override_kwargs(pipeline.transformer, image_rotary_emb = image_rotary_emb, attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": self.attention_mask_stage}), \
            #     crop_output_frames(pipeline.transformer, keep=F, out_index=0, dim=1):
            #     video = pipeline(**generation_kwargs).frames[0]
            return [VideoArtifact(value=video)]
        else:
            raise ValueError(f"Stage {stage} not supported. Only 1 and 2 are supported.")

    def implement_rope_custom(self, 
                              rope_enumeration_method: str,
                              vidpart_latent_height: int,
                              vidpart_latent_width: int,
                              vidpart_latent_frame_length: int,
                              final_frame_length: int,
                              vae_spatial_scale_factor: int,
                              patch_size: int,
                              patch_size_t: Union[int, None],
                              device: Union[torch.device, str],
                              rope_base_height: int,
                              rope_base_width: int,
                              offset_w_multiplier: int, 
                              offset_h_multiplier: int,
                              number_of_subject: int):

        "vidpart_latent_height is H"
        "vidpart_latent_width is W"
        "vae_spatial_scale_factor is VAE_SPATIAL_SCALE_FACTOR"
        "vidpart_latent_frame_length is F"
        if rope_enumeration_method == "normal":
            image_rotary_emb = (
                prepare_rotary_positional_embeddings_cogvideox(
                    height=vidpart_latent_height * vae_spatial_scale_factor,
                    width=vidpart_latent_width * vae_spatial_scale_factor,
                    num_frames=final_frame_length,
                    vae_scale_factor_spatial=vae_spatial_scale_factor,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    attention_head_dim=self.transformer_config.attention_head_dim,
                    device=device,
                    base_height=rope_base_height,
                    base_width=rope_base_width,
                )
                if self.transformer_config.use_rotary_positional_embeddings
                else None
            )
        elif rope_enumeration_method == "separate_every_modality_but_video_temporal_same":
            vid_part_rotary_emb = (
                    prepare_rotary_positional_embeddings_cogvideox(
                        height=vidpart_latent_height * vae_spatial_scale_factor,
                        width=vidpart_latent_width * vae_spatial_scale_factor,
                        num_frames=vidpart_latent_frame_length,
                        vae_scale_factor_spatial=vae_spatial_scale_factor,
                        patch_size=patch_size,
                        patch_size_t=patch_size_t,
                        attention_head_dim=self.transformer_config.attention_head_dim,
                        device=device,
                        base_height=rope_base_height,
                        base_width=rope_base_width,
                    )
                    if self.transformer_config.use_rotary_positional_embeddings
                    else None
                )
            sketch_part_rotary_emb = (
                prepare_rotary_positional_embeddings_cogvideox(
                    height=vidpart_latent_height * vae_spatial_scale_factor,
                    width=vidpart_latent_width * vae_spatial_scale_factor,
                    num_frames=vidpart_latent_frame_length,
                    vae_scale_factor_spatial=vae_spatial_scale_factor,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    attention_head_dim=self.transformer_config.attention_head_dim,
                    device=device,
                    base_height=rope_base_height,
                    base_width=rope_base_width,
                    offset_w = offset_w_multiplier, 
                    offset_h = offset_h_multiplier,
                )
                if self.transformer_config.use_rotary_positional_embeddings
                else None
            )
            subject_part_rotary_emb = (
                prepare_rotary_positional_embeddings_cogvideox(
                    height=vidpart_latent_height * vae_spatial_scale_factor,
                    width=vidpart_latent_width * vae_spatial_scale_factor,
                    num_frames=1+number_of_subject,
                    vae_scale_factor_spatial=vae_spatial_scale_factor,
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    attention_head_dim=self.transformer_config.attention_head_dim,
                    device=device,
                    base_height=rope_base_height,
                    base_width=rope_base_width,
                    negative_temporal_embedding = True,
                    offset_w=3 * offset_w_multiplier,
                    offset_h=3 * offset_h_multiplier,
                )
                if self.transformer_config.use_rotary_positional_embeddings
                else None
            )
            print("Vid part rotary emb: ", vid_part_rotary_emb[0].shape)
            print("Sketch part rotary emb: ", vid_part_rotary_emb[0].shape)
            print("Subject part rotary emb: ", subject_part_rotary_emb[0].shape)

            image_rotary_emb = concatenate_freqs_cos_freqs_sin_temporally(
                [vid_part_rotary_emb[0], sketch_part_rotary_emb[0], subject_part_rotary_emb[0]],
                [vid_part_rotary_emb[1], sketch_part_rotary_emb[1], subject_part_rotary_emb[1]])
        else:
            raise ValueError(f"Invalid rope_enumeration_method. Please choose a valid method. You input: {rope_enumeration_method}")
        
        return image_rotary_emb
    
    def _recover_token_spans(self, 
                             stage: int, 
                             F: int, 
                             H: int,
                             W: int,
                             patch_size: int,
                             number_of_subject: int,
                             device: torch.device) -> Dict[List[Tuple[int, int]], List[int]]:
        print("Token spans modification ")
        Hp = H//2
        Wp = W//2
        span_length = Hp * Wp * F
        firstEnd = 226 + span_length
        secondEnd = 226 + span_length*2
        thirdEnd = 226 + span_length*3
        firstReferenceEndFromThird = thirdEnd + 1350
        firstReferenceEndFromSecond = secondEnd + 1350
        token_spans = {
            "TEXT":        (0, 226),
            "MAIN":    (226, firstEnd),
            "SKETCH":  (firstEnd, secondEnd),
            "REF_0":       (secondEnd, firstReferenceEndFromSecond),
        }
        for i in range(number_of_subject):
            token_spans[f"REF_{i+1}"] = (firstReferenceEndFromSecond + i*1350, firstReferenceEndFromSecond + (i+1)*1350)
        return token_spans

    def expand_token_spans_with_attention_pixel_pack_ontherun(
        self,
        token_spans: Dict[str, Tuple[int, int]],
        *,
        attn_ids: Union[np.ndarray, torch.Tensor],
        F: int,
        H: int,
        W: int,
        device: torch.device,
        allowed_ids: Iterable[int] = (0, 1, 2, 3),
        strict_shape: bool = True,
        strict_values: bool = True,
        main_key: str = "MAIN",
    ) -> Dict[str, object]:
        """
        Adds:
        token_spans["PIXEL_PACK"] = {...}

        Does NOT materialize PIXELS_REF_0..3 arrays up front.
        """
        if main_key not in token_spans:
            raise KeyError(f"{main_key!r} not found in token_spans keys={list(token_spans.keys())}")

        main_start, main_end = token_spans[main_key]
        

        pixel_pack = self.build_attention_pixel_pack_from_attn_ids(
            attn_ids=attn_ids,
            F=F,
            H=H,
            W=W,
            device=device,
            allowed_ids=allowed_ids,
            strict_shape=strict_shape,
            strict_values=strict_values,
        )

        expected_main_len = int(pixel_pack["N"])
        main_len = main_end - main_start
        if main_len != expected_main_len:
            raise RuntimeError(
                f"MAIN span length mismatch: (end-start)={main_len} != N={expected_main_len} "
                f"mask={pixel_pack.get('mask_path')}"
            )

        out: Dict[str, object] = dict(token_spans)
        out["PIXEL_PACK"] = pixel_pack
        return out
    def build_attention_pixel_pack_from_attn_ids(
        self,
        *,
        attn_ids: Union[np.ndarray, torch.Tensor],
        F: int,
        H: int,
        W: int,
        device: torch.device,
        allowed_ids: Iterable[int] = (0, 1, 2, 3),
        strict_shape: bool = True,
        strict_values: bool = True,
    ) -> Dict[str, object]:
        """
        Same output contract as build_attention_pixel_pack(), but reads mask from
        an in-memory attention-id mask instead of loading NPZ.

        Expected attn_ids shape is (F, Hp, Wp) where Hp = H//2, Wp = W//2
        (matching your existing code's convention).
        """

        # Accept torch.Tensor too (common if you already have it on GPU/CPU)
        if isinstance(attn_ids, torch.Tensor):
            mask_arr = attn_ids.detach().to("cpu").numpy()
        else:
            mask_arr = attn_ids

        mask_arr = np.asarray(mask_arr)

        # Ensure integer type for labels
        if mask_arr.dtype.kind not in ("i", "u"):
            mask_arr = mask_arr.astype(np.int64, copy=False)
        else:
            mask_arr = mask_arr.astype(np.int64, copy=False)

        Hp = H // 2
        Wp = W // 2
        expected_shape = (F, Hp, Wp)

        if strict_shape and tuple(mask_arr.shape) != expected_shape:
            raise RuntimeError(
                f"attention mask shape {tuple(mask_arr.shape)} != expected {expected_shape} "
            )

        if not strict_shape:
            if mask_arr.ndim != 3:
                raise RuntimeError(f"Expected 3D mask (F,Hp,Wp), got shape {tuple(mask_arr.shape)}")
            F, Hp, Wp = map(int, mask_arr.shape)

        allowed_ids = tuple(int(x) for x in allowed_ids)

        if strict_values:
            uniq = np.unique(mask_arr)
            bad = [int(x) for x in uniq if int(x) not in allowed_ids]
            if bad:
                raise RuntimeError(
                    f"attention mask has values outside {allowed_ids}: bad={bad[:20]} "
                )

        # Flatten labels
        flat_labels = mask_arr.reshape(-1)  # (N,)
        N = int(flat_labels.size)
        if N == 0:
            return {
                "F": int(F),
                "Hp": int(Hp),
                "Wp": int(Wp),
                "N": 0,
                "K": 0,
                "allowed_ids": allowed_ids,
                "sorted_local_idx": torch.empty((0,), dtype=torch.int32, device=device),
                "ptr": torch.zeros((2,), dtype=torch.int32, device=device),  # K=0 => K+2=2
                "mask_source": "in_memory_attn_ids",
            }

        K = int(flat_labels.max())

        # local indices 0..N-1
        local_idx = np.arange(N, dtype=np.int32)

        # stable sort by label => groups contiguous
        order = np.argsort(flat_labels, kind="stable")
        sorted_labels = flat_labels[order]
        sorted_local_idx = local_idx[order]

        # ptr for labels 0..K: ptr length K+2
        ptr_np = np.searchsorted(sorted_labels, np.arange(K + 2), side="left").astype(np.int32)

        pixel_pack = {
            "F": int(F),
            "Hp": int(Hp),
            "Wp": int(Wp),
            "N": int(N),
            "K": int(K),
            "allowed_ids": allowed_ids,
            "sorted_local_idx": torch.from_numpy(sorted_local_idx).to(device=device, non_blocking=True),
            "ptr": torch.from_numpy(ptr_np).to(device=device, non_blocking=True),
            "mask_source": "in_memory_attn_ids",
        }
        return pixel_pack

    @staticmethod
    def _pad_frames(latents: torch.Tensor, patch_size_t: int) -> torch.Tensor:
        num_frames = latents.size(1)
        additional_frames = patch_size_t - (num_frames % patch_size_t)
        if additional_frames > 0:
            last_frame = latents[:, -1:]
            padding_frames = last_frame.expand(-1, additional_frames, -1, -1, -1)
            latents = torch.cat([latents, padding_frames], dim=1)
        return latents