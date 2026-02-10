import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
    
import functools
from typing import List, Optional, Tuple, Union, Dict, Iterable
import argparse
import time
import torch
import os
import json

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Prefer Flash Attention / efficient SDP
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from diffusers import DiffusionPipeline, CogVideoXPipeline, CogVideoXTransformer3DModel, \
    CogVideoXDDIMScheduler

from timecolor.models.cogvideox.temporal_pipeline import CogVideoXTemporalPipeline
from timecolor.models.cogvideox.base_specification import _load_npz_mask, \
    make_attention_id_mask_from_mask_mp4, make_attention_id_mask_from_mask_npz
import numpy as np

from diffusers.utils import export_to_video
from PIL.Image import Image

from pathlib import Path
from timecolor.models.cogvideox.utils import (
    prepare_rotary_positional_embeddings_cogvideox,
    concatenate_freqs_cos_freqs_sin_temporally,
)
from timecolor.utils import _enable_vae_memory_optimizations, get_non_null_items
from diffusers.utils import load_image, load_video


def parallelize_transformer(pipe: DiffusionPipeline):
    """
    Single-GPU stub: keeps the original transformer.forward unchanged.

    (Your original code overwrote forward() to do CFG-batch sharding + all_gather.
     On single GPU, that logic is unnecessary.)
    """
    return


def implement_rope_custom(
    transformer_config,
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
    number_of_subject: int,
):
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
                attention_head_dim=transformer_config.attention_head_dim,
                device=device,
                base_height=rope_base_height,
                base_width=rope_base_width,
            )
            if transformer_config.use_rotary_positional_embeddings
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
                attention_head_dim=transformer_config.attention_head_dim,
                device=device,
                base_height=rope_base_height,
                base_width=rope_base_width,
            )
            if transformer_config.use_rotary_positional_embeddings
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
                attention_head_dim=transformer_config.attention_head_dim,
                device=device,
                base_height=rope_base_height,
                base_width=rope_base_width,
                offset_w=offset_w_multiplier,
                offset_h=offset_h_multiplier,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        subject_part_rotary_emb = (
            prepare_rotary_positional_embeddings_cogvideox(
                height=vidpart_latent_height * vae_spatial_scale_factor,
                width=vidpart_latent_width * vae_spatial_scale_factor,
                num_frames=1 + number_of_subject,
                vae_scale_factor_spatial=vae_spatial_scale_factor,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                attention_head_dim=transformer_config.attention_head_dim,
                device=device,
                base_height=rope_base_height,
                base_width=rope_base_width,
                negative_temporal_embedding=True,
                offset_w=3 * offset_w_multiplier,
                offset_h=3 * offset_h_multiplier,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        print("Vid part rotary emb: ", vid_part_rotary_emb[0].shape)
        print("Sketch part rotary emb: ", vid_part_rotary_emb[0].shape)
        print("Subject part rotary emb: ", subject_part_rotary_emb[0].shape)

        image_rotary_emb = concatenate_freqs_cos_freqs_sin_temporally(
            [vid_part_rotary_emb[0], sketch_part_rotary_emb[0], subject_part_rotary_emb[0]],
            [vid_part_rotary_emb[1], sketch_part_rotary_emb[1], subject_part_rotary_emb[1]],
        )
    else:
        raise ValueError(
            f"Invalid rope_enumeration_method. Please choose a valid method. You input: {rope_enumeration_method}"
        )

    return image_rotary_emb


def _recover_token_spans(F: int, H: int, W: int, number_of_subject: int):
    print("Token spans modification ")
    Hp = H // 2
    Wp = W // 2
    span_length = Hp * Wp * F
    firstEnd = 226 + span_length
    secondEnd = 226 + span_length * 2
    thirdEnd = 226 + span_length * 3
    firstReferenceEndFromThird = thirdEnd + 1350
    firstReferenceEndFromSecond = secondEnd + 1350
    token_spans = {
        "TEXT": (0, 226),
        "MAIN": (226, firstEnd),
        "SKETCH": (firstEnd, secondEnd),
        "REF_0": (secondEnd, firstReferenceEndFromSecond),
    }
    for i in range(number_of_subject):
        token_spans[f"REF_{i+1}"] = (
            firstReferenceEndFromSecond + i * 1350,
            firstReferenceEndFromSecond + (i + 1) * 1350,
        )
    return token_spans


def expand_token_spans_with_attention_pixel_pack_ontherun(
    token_spans: Dict[str, Tuple[int, int]],
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

    pixel_pack = build_attention_pixel_pack_from_attn_ids(
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

    if isinstance(attn_ids, torch.Tensor):
        mask_arr = attn_ids.detach().to("cpu").numpy()
    else:
        mask_arr = attn_ids

    mask_arr = np.asarray(mask_arr)

    if mask_arr.dtype.kind not in ("i", "u"):
        mask_arr = mask_arr.astype(np.int64, copy=False)
    else:
        mask_arr = mask_arr.astype(np.int64, copy=False)

    Hp = H // 2
    Wp = W // 2
    expected_shape = (F, Hp, Wp)

    if strict_shape and tuple(mask_arr.shape) != expected_shape:
        raise RuntimeError(f"attention mask shape {tuple(mask_arr.shape)} != expected {expected_shape} ")

    if not strict_shape:
        if mask_arr.ndim != 3:
            raise RuntimeError(f"Expected 3D mask (F,Hp,Wp), got shape {tuple(mask_arr.shape)}")
        F, Hp, Wp = map(int, mask_arr.shape)

    allowed_ids = tuple(int(x) for x in allowed_ids)

    if strict_values:
        uniq = np.unique(mask_arr)
        bad = [int(x) for x in uniq if int(x) not in allowed_ids]
        if bad:
            raise RuntimeError(f"attention mask has values outside {allowed_ids}: bad={bad[:20]} ")

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
            "ptr": torch.zeros((2,), dtype=torch.int32, device=device),
            "mask_source": "in_memory_attn_ids",
        }

    K = int(flat_labels.max())

    local_idx = np.arange(N, dtype=np.int32)

    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    sorted_local_idx = local_idx[order]

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


def run_sample(
    model_path: Path,
    transformer_path: Path,
    cache_dir: Path,
    sketch_control_path: str,
    prompt: str,
    height: Optional[int] = 480,
    width: Optional[int] = 720,
    rope_enumeration_method: str = "INVALID",
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    identity_mask_npz_path: Optional[np.ndarray] = None,
    generator: Optional[torch.Generator] = None,
    reference_paths: Optional[List[str]] = None,
    guidance_scale=3,
    enable_slicing=True,
    enable_tiling=True,
    enable_cpu_offload=True,
    dtype=torch.bfloat16,
    out_path=None,
    fps=15,
    local_rank=0,
):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(transformer_path, subfolder="scheduler")

    number_of_subject = len(reference_paths) - 1

    sketch_control = load_video(sketch_control_path)
    if reference_paths:
        references = []
        for i in reference_paths:
            one_ref = load_image(i)
            references.append(one_ref)

    identity_mask_npz = None
    if number_of_subject > 0:
        assert identity_mask_npz_path is not None
        identity_mask_npz = _load_npz_mask(identity_mask_npz_path)

    pipeline = CogVideoXTemporalPipeline.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=dtype,
    )
    _enable_vae_memory_optimizations(pipeline.vae, enable_slicing, enable_tiling)

    # Single GPU: keep behavior consistent with your flags.
    if enable_cpu_offload and device.type == "cuda":
        pipeline.enable_model_cpu_offload(gpu_id=local_rank)
    else:
        pipeline = pipeline.to(device)

    pipeline.text_encoder.to(dtype)
    pipeline.vae.to(dtype)

    vae_config = pipeline.vae.config
    transformer_config = pipeline.transformer.config

    parallelize_transformer(pipeline)
    print("Initialized cogvideox temporal pipeline")

    if number_of_subject == 0:
        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device

            single_ref = pipeline.video_processor.preprocess(
                references[0], height=height, width=width
            )
            single_ref = single_ref.to(device=device, dtype=dtype)
            sketch_control = pipeline.video_processor.preprocess_video(sketch_control, height=height, width=width)
            sketch_control = sketch_control.to(device=device, dtype=dtype)

            single_ref = single_ref.unsqueeze(2)  # B C 1 H W
            single_ref = pipeline.vae.encode(single_ref).latent_dist.sample(
                generator=generator
            )
            sketch_control = pipeline.vae.encode(sketch_control).latent_dist.sample(generator=generator)

        B, C, F, H, W = sketch_control.shape

        single_ref = vae_config.scaling_factor * single_ref
        sketch_control = vae_config.scaling_factor * sketch_control

        patch_size = transformer_config.patch_size
        patch_size_t = getattr(transformer_config, "patch_size_t", None)
        VAE_SPATIAL_SCALE_FACTOR = 8
        rope_base_height = transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
        rope_base_width = transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
        final_frame_length = 2 * F + 1 + number_of_subject
        OFFSET_W_MULTIPLIER = (W * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
        OFFSET_H_MULTIPLIER = (H * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)

        image_rotary_emb = implement_rope_custom(
            transformer_config=transformer_config,
            rope_enumeration_method=rope_enumeration_method,
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
            number_of_subject=number_of_subject,
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

        token_spans = _recover_token_spans(F=F, H=H, W=W, number_of_subject=number_of_subject)

        single_ref = single_ref.permute(0, 2, 1, 3, 4)
        sketch_control = sketch_control.permute(0, 2, 1, 3, 4)

        video = pipeline(
            **generation_kwargs,
            sketch_control_hiddenstates=sketch_control,
            subjects_refs_hiddenstates=single_ref,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": 3},
            keep_frames=F,
            out_index=0,
            out_dim=1,
        ).frames[0]

    else:
        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device
            sketch_control = pipeline.video_processor.preprocess_video(sketch_control, height=height, width=width)
            sketch_control = sketch_control.to(device=device, dtype=dtype)
            references = [pipeline.video_processor.preprocess_video(ref, height=height, width=width) for ref in references]
            references = [ref.to(device=device, dtype=dtype) for ref in references]

            # NOTE: Keeping your original structure (this loop does not modify the list elements in-place).
            for ref in references:
                ref = ref.unsqueeze(2)  # B C 1 H W

            sketch_control = pipeline.vae.encode(sketch_control).latent_dist.sample(generator=generator)
            for i, ref in enumerate(references):
                references[i] = pipeline.vae.encode(ref).latent_dist.sample(generator=generator)

            sketch_control = vae_config.scaling_factor * sketch_control
            references = [vae_config.scaling_factor * ref for ref in references]

        B, C, F, H, W = sketch_control.shape
        patch_size = transformer_config.patch_size
        patch_size_t = getattr(transformer_config, "patch_size_t", None)
        VAE_SPATIAL_SCALE_FACTOR = 8
        rope_base_height = transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
        rope_base_width = transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
        final_frame_length = 2 * F + 1 + number_of_subject
        OFFSET_W_MULTIPLIER = (W * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)
        OFFSET_H_MULTIPLIER = (H * VAE_SPATIAL_SCALE_FACTOR) // (VAE_SPATIAL_SCALE_FACTOR * patch_size)

        image_rotary_emb = implement_rope_custom(
            transformer_config=transformer_config,
            rope_enumeration_method=rope_enumeration_method,
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
            number_of_subject=number_of_subject,
        )

        token_spans = _recover_token_spans(F=F, H=H, W=W, number_of_subject=number_of_subject)

        if identity_mask_npz is not None:
            real_ids, latent_ids, attn_ids = make_attention_id_mask_from_mask_npz(identity_mask_npz)

        token_spans = expand_token_spans_with_attention_pixel_pack_ontherun(
            token_spans=token_spans,
            attn_ids=attn_ids,
            F=F,
            H=H,
            W=W,
            device=device,
            allowed_ids=[i for i in range(len(references))],
            strict_shape=True,
            strict_values=True,
            main_key="MAIN",
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
        }
        generation_kwargs = get_non_null_items(generation_kwargs)

        subjects_refs = torch.concat(references, dim=2)  # B, C, N, H, W

        subjects_refs = subjects_refs.permute(0, 2, 1, 3, 4)
        sketch_control = sketch_control.permute(0, 2, 1, 3, 4)

        video = pipeline(
            **generation_kwargs,
            sketch_control_hiddenstates=sketch_control,
            subjects_refs_hiddenstates=subjects_refs,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs={"mask_tokens_range": token_spans, "attention_mask_level": 4},
            keep_frames=F,
            out_index=0,
            out_dim=1,
        ).frames[0]

    # Single GPU => rank is always 0
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        saved = export_to_video(video, str(out_path), fps=int(fps))
        print("Saved video to", saved)
    return video


def parse_common_args():
    p = argparse.ArgumentParser(description="CLI for run_sample(...)")
    p.add_argument("--model-path", type=str, default="THUDM/CogVideoX-5b")
    p.add_argument("--cache-dir", type=str, default="./checkpoint/")
    p.add_argument("--transformer-path", type=str, default="./checkpoint/TimeColor-final/model_weights")
    p.add_argument("--seed", type=int, default=0, help="Seed used to create torch.Generator (optional)")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--work_json_path", type=str, default="./examples/inference_samples.json")

    args = p.parse_args()
    args.enable_slicing = True
    args.enable_tiling = True
    args.enable_cpu_offload = True
    args.rope_enumeration_method = "separate_every_modality_but_video_temporal_same"

    return args


def setup_single_gpu():
    local_rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_rank0 = True
    return local_rank, device, is_rank0

def read_work_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    return obj["data"]  # list of dicts



if __name__ == "__main__":
    args = parse_common_args()
    local_rank, device, is_rank0 = setup_single_gpu()
    dtype = torch.bfloat16

    work_json = read_work_json(args.work_json_path)

    for job in work_json:
        height = job["height"]
        width = job["width"]
        num_inference_steps = job["num_inference_steps"]
        reference_paths = job["reference_paths"]
        identity_mask_npz_path = None
        if len(reference_paths) > 1:
            identity_mask_npz_path = job["identity_mask_npz_path"]
        num_frames = job["num_frames"]
        fps = job["frame_rate"]
        prompt = job["caption"]
        custom_output_name = job["custom_output_name"]
        sketch_control_path = job["sketch_control_path"]
        generator = torch.Generator(device=device).manual_seed(args.seed)

        run_kwargs = dict(
            model_path=args.model_path,
            transformer_path=(args.transformer_path if args.transformer_path is not None else args.model_path),
            cache_dir=args.cache_dir,
            sketch_control_path=sketch_control_path,
            prompt=prompt,
            height=height,
            width=width,
            rope_enumeration_method=args.rope_enumeration_method,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            identity_mask_npz_path=identity_mask_npz_path,
            generator=generator,
            reference_paths=reference_paths,
            guidance_scale=args.guidance_scale,
            enable_slicing=args.enable_slicing,
            enable_tiling=args.enable_tiling,
            enable_cpu_offload=args.enable_cpu_offload,
            dtype=dtype,
            out_path=custom_output_name if is_rank0 else None,
            fps=fps,
            local_rank=local_rank,
        )

        video = run_sample(**run_kwargs)
