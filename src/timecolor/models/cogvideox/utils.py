from typing import Optional, Tuple

import torch
import numpy as np
from typing import Union


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    """Purpose – Given a source grid size (h, w) (e.g., latent space height/width in patches) and a target “canvas” (tgt_height, tgt_width), it tells you how to letter‑box:
    how much to shrink (preserve aspect ratio), where to crop/pad centrally"""
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    negative_temporal_embedding: bool = False,
    offset_h: int = 0,
    offset_w: int = 0,
    offset_t: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords 
        grid_size_h, grid_size_w = grid_size 
        grid_h = torch.linspace(
            start[0] + offset_h, stop[0] * (grid_size_h - 1) / grid_size_h + offset_h, grid_size_h, device=device, dtype=torch.float32
        )
        grid_w = torch.linspace(
            start[1] + offset_w, stop[1] * (grid_size_w - 1) / grid_size_w + offset_w, grid_size_w, device=device, dtype=torch.float32
        )
        if negative_temporal_embedding:
            grid_t = torch.linspace(
                -1 + offset_t,
                -temporal_size + offset_t,
                temporal_size,
                device=device,
                dtype=torch.float32,
            )
        else:
            grid_t = torch.linspace(
                0+offset_t, temporal_size * (temporal_size - 1) / temporal_size + offset_t, temporal_size, device=device, dtype=torch.float32
            )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4 #split 25% for temporal
    dim_h = embed_dim // 8 * 3 # split 37.5% for height
    dim_w = embed_dim // 8 * 3 # split 37.5% for width

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def prepare_rotary_positional_embeddings_cogvideox(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
    negative_temporal_embedding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    
    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    #embed dim equals to attention_head_dim since 
    # must decide which channel pairs encode which axis
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        negative_temporal_embedding=negative_temporal_embedding,
    )
    

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def prepare_rotary_positional_embeddings_cogvideox(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
    offset_h: int = 0,         # shift (in latent patches) applied to H axis
    offset_w: int = 0,         # shift applied to W axis
    offset_t: int = 0,         # shift applied to T axis
    negative_temporal_embedding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size) #480 // (8*2) = 30
    grid_width = width // (vae_scale_factor_spatial * patch_size) #720 // (8*2) = 45
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size) #30
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size) #45

    
    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    #embed dim equals to attention_head_dim since 
    # must decide which channel pairs encode which axis
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        negative_temporal_embedding=negative_temporal_embedding,
        offset_w=offset_w,
        offset_h=offset_h,
        offset_t=offset_t,
    )
    

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def concatenate_freqs_cos_freqs_sin_temporally(freqs_cosines, freqs_sines):
    """
    Concatenate the cosine and sine frequency tensors temporally.
    This is necessary for the temporal concatenation application of rotary positional embeddings.

    freqs_cosines and freqs_sines is in list of tensors
    """
    assert len(freqs_cosines) == len(freqs_sines), "Cosine and sine tensors must have the same shape."

    #complete the rest
    concatenated_cos = torch.cat(freqs_cosines, dim=0)
    concatenated_sin = torch.cat(freqs_sines,   dim=0)
    return concatenated_cos, concatenated_sin

import io, pathlib as pl, torch, gzip  # igzip optional

MAGIC_GZIP = b"\x1f\x8b"
MAGIC_ZIP  = b"PK"

try:
    import deflate                         # fast C routine
    fast_decompress = deflate.gzip_decompress
    print("Deflate imported")
except ImportError:
    fast_decompress = gzip.decompress         # fallback
    print("Deflate not imported, using gzip fallback")


def smart_load_pt(path: pl.Path, map_location="cpu"):
    if isinstance(path, str):
        path = pl.Path(path)
    with path.open("rb") as fh:
        magic = fh.read(2)
        fh.seek(0)                     # rewind

        if magic == MAGIC_GZIP:
            # --- fastest: igzip; fall back to std‑lib gzip if unavailable
            raw = fh.read()
            data = fast_decompress(raw)
            #print("enter A")
            return torch.load(io.BytesIO(data), map_location=map_location)

        elif magic == MAGIC_ZIP:
            # already a ZIP, let torch handle it
            #print("enter B")
            return torch.load(fh, map_location=map_location)

        else:
            raise RuntimeError(f"Unknown .pt variant: starts with {magic!r}")