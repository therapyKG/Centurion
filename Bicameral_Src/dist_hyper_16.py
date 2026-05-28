#MEM Efficient and trains fast
import subprocess, sys
# Ensure required packages are present
for pkg in ["datasets", "sentencepiece", "transformers", "huggingface_hub"]:
    try: __import__(pkg)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math, time, gc, json, os, bisect
from dataclasses import dataclass, asdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import LlamaTokenizerFast
import sentencepiece as spm

# =============================================================================
# Distributed setup
# =============================================================================
#
# We support two run modes:
#   1. Single-process (no torchrun, no env vars): runs as a standard single-GPU
#      script. dist is not initialized.
#   2. Multi-process via torchrun: torchrun sets RANK, WORLD_SIZE, LOCAL_RANK
#      env vars and launches one process per GPU. We initialize NCCL backend
#      and use DistributedDataParallel.
#
# Detect the run mode early so the tokenizer download only happens on rank 0
# (otherwise multiple processes race on the same file).

def _is_torchrun():
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))

def _setup_distributed():
    """Initialize NCCL backend if running under torchrun.

    Returns (rank, local_rank, world_size, is_distributed).
    Each rank pins itself to its assigned GPU via torch.cuda.set_device.
    """
    if _is_torchrun():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False

def _cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

def _is_main_process():
    """True if rank 0, regardless of whether dist is initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def _unwrap_model(model):
    """Strip DDP and torch.compile wrappers to access the underlying nn.Module.

    DDP wraps the model in DistributedDataParallel.module. torch.compile wraps
    in OptimizedModule._orig_mod. We unwrap both, since architecture-touching
    code paths (diagnostics, parameter iteration, tokenizer setup) need to see
    the original module, not the wrapper.
    """
    m = model
    while True:
        if hasattr(m, "module") and isinstance(m, nn.parallel.DistributedDataParallel):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            return m

# Tokenizer download: only rank 0 downloads, others wait at the barrier.
# Outside torchrun, this just runs once on the single process.
if _is_main_process() or not _is_torchrun():
    hf_hub_download(
        repo_id="openlm-research/open_llama_3b",
        filename="tokenizer.model",
        local_dir="."
    )
# Note: barrier happens later, after init_process_group is called.


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Model shape (LLaMA-3.2-1B-like)
    n_layers: int = 16
    d_model: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ffn: int = 8192
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # Hyper config
    latent_dim: int = 1024
    generator_hidden: int = 2048
    generator_layers: int = 2
    block_k: int = 128
    block_n: int = 128

    # Training
    batch_size: int = 96
    grad_accum_steps: int = 1
    max_steps: int = 40000
    warmup_steps: int = 400
    max_lr: float = 1e-3
    min_lr: float = 3e-5
    z_max_lr: float = 1e-3
    z_min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gen_lr_scale: float = 1.0
    gen_lr_flat: bool = False

    # Data paths
    data_dir: str = "/workspace/data/"
    checkpoint_dir: str = "checkpoints/hyper64"

    # Dataset selection
    dataset: str = "fineweb"

    # Eval & logging
    eval_interval: int = 100
    eval_steps: int = 20
    log_interval: int = 50
    timing_warmup: int = 5
    save_interval: int = 5000

    # What to run
    use_compile: bool = True
    run_standard: bool = False
    run_hyper: bool = True
    resume_from: str = ""

    @property
    def d_head(self): return self.d_model // self.n_heads
    @property
    def effective_batch(self): return self.batch_size * self.grad_accum_steps

    def validate_alignment(self):
        """
        Verify all projection dimensions are exact multiples of block_k/block_n.

        WHY THIS MATTERS:
        When dims aren't aligned, tile assembly requires a slice ([:in_f, :out_f])
        that produces non-contiguous tensors. This causes:
          1. Extra copies before cuBLAS matmuls
          2. RMS(tiles) != RMS(sliced_W), breaking the normalization invariant
          3. Non-deterministic Triton autotuning on the irregular shapes

        Enforcing alignment makes the slice a no-op, the reshape output contiguous,
        and RMS computation on tiles exactly equivalent to RMS on the assembled W.
        """
        bk, bn = self.block_k, self.block_n
        dims = {
            "d_model":         self.d_model,
            "d_ffn":           self.d_ffn,
            "n_kv_heads*d_head": self.n_kv_heads * self.d_head,
        }
        for name, dim in dims.items():
            if dim % bk != 0:
                raise ValueError(
                    f"{name}={dim} is not a multiple of block_k={bk}. "
                    f"Nearest aligned values: {(dim // bk) * bk} or {((dim // bk) + 1) * bk}"
                )
            if dim % bn != 0:
                raise ValueError(
                    f"{name}={dim} is not a multiple of block_n={bn}. "
                    f"Nearest aligned values: {(dim // bn) * bn} or {((dim // bn) + 1) * bn}"
                )

# =============================================================================
# GPU detection
# =============================================================================

def detect_gpu(local_rank=0, verbose=True):
    """Detect GPU properties for the current rank's GPU. local_rank specifies
    which GPU to query — under DDP each rank set torch.cuda.set_device(local_rank)
    in _setup_distributed, and we report on that GPU here for accurate logging."""
    assert torch.cuda.is_available(), "No CUDA GPU found!"
    name = torch.cuda.get_device_name(local_rank)
    vram = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    bf16_ok = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    if verbose:
        print(f"GPU: {name} ({vram:.1f} GB) | {'bf16' if bf16_ok else 'fp16'} | PyTorch {torch.__version__}")
    return dtype

# =============================================================================
# Shared components
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :])
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :])
    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class SwishBeta(nn.Module):
    def __init__(self, init_beta=1.0, learnable=False):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor([init_beta]))
        else:
            self.register_buffer('beta', torch.tensor([init_beta]))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

# =============================================================================
# Tile generator (unchanged from V2)
# =============================================================================

class TileGenerator(nn.Module):
    """MLP-only: tile = MLP(z)"""
    def __init__(self, latent_dim, hidden_dim, block_k, block_n, n_layers=2):
        super().__init__()
        tile_size = block_k * block_n
        self.block_k, self.block_n, self.latent_dim = block_k, block_n, latent_dim
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim, bias=False), SwishBeta(init_beta=1.0, learnable=True)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, tile_size, bias=False))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=0.12)

    def forward(self, z):
        orig_shape = z.shape[:-1]
        return self.net(z.reshape(-1, self.latent_dim)).view(*orig_shape, self.block_k, self.block_n)

# =============================================================================
# Tile assembly — hidden from torch.compile via allow_in_graph
# =============================================================================
#
# KEY CHANGE vs V2: The permute → reshape → slice chain is wrapped in
# allow_in_graph so Dynamo treats it as an opaque leaf node. This means:
#   - Triton never sees the irregular intermediate shapes
#   - The compiler only traces the clean x @ W matmul that follows
#   - No Triton config explosion when d_model or batch changes
#
# The function runs eagerly inside the compiled graph. Since it's just
# memory layout manipulation (no compute), the overhead is negligible.
#
# We also guarantee contiguity of the output by making the slice a no-op
# via dimension alignment (see Config.validate_alignment).

@torch.compiler.allow_in_graph
def _assemble_weight(tiles, scale, in_f, out_f):
    """
    Assemble a dense weight matrix from tiles and apply scale.

    tiles: (num_k, num_n, block_k, block_n) — raw generator output (post per-column RMS)
    scale: scalar or (out_f,) — per-column scale = scale_init * exp(log_scale).
           Broadcasting takes care of either case: (in_f, out_f) * (out_f,) gives
           column-wise scaling; (in_f, out_f) * scalar gives global scaling.
    in_f:  int — input features (== num_k * block_k when aligned)
    out_f: int — output features (== num_n * block_n when aligned)

    Returns: (in_f, out_f) contiguous weight matrix
    """
    # permute: (num_k, num_n, bk, bn) → (num_k, bk, num_n, bn)
    # reshape: → (padded_in, padded_out)
    # When aligned, padded_in == in_f and padded_out == out_f, so slice is no-op
    num_k, num_n, bk, bn = tiles.shape
    W = tiles.permute(0, 2, 1, 3).reshape(num_k * bk, num_n * bn)
    W = W[:in_f, :out_f]
    # .contiguous() is free when aligned (already contiguous), safety net otherwise
    return (W * scale).contiguous()


# =============================================================================
# HyperLinear
# =============================================================================
#
# KEY CHANGES vs V2:
#   1. RMS computed on tiles directly (not on assembled W).
#      Mathematically identical when dimensions are block-aligned because
#      reshape preserves elements → mean(tiles²) == mean(W²).
#      Avoids materializing W just for normalization.
#
#   2. Uses x @ W instead of F.linear(x, W.t()).
#      F.linear(x, weight) expects weight in (out, in) layout and does x @ weight.T.
#      V2 assembled W as (in, out) then transposed it — double layout change.
#      x @ W with W in (in, out) is the natural layout. Eliminates the transpose
#      and gives cuBLAS the matrix in the layout it actually uses.
#
#   3. Assembly goes through _assemble_weight (allow_in_graph).
#      The compiler sees: tiles → opaque_fn → W → matmul. Clean.

class HyperLinear(nn.Module):
    """Weight is generated tiles, per-output-column RMS-normalized, scaled by scale_init * exp(log_scale).

    log_scale is a per-output-column parameter (shape (out_features,)), giving the
    optimizer a degree of freedom to set magnitudes of individual output channels of
    W independently. This is one rigidity restored vs. the all-equal-column
    normalization that per-column RMS would otherwise impose: RMS still constrains
    each column to unit norm relative to its peers, then log_scale rescales each
    column independently. Without this, each HyperLinear can only set a single
    global magnitude for all of W's columns, which is a real expressivity cost.

    The cost is small: out_features parameters per HyperLinear, totaling
    sum(out_features for HL in model) ~= O(L * d_model + L * d_ffn) = ~L * 5d
    extra params (e.g. for L=16, d=2048: ~160k extra params, negligible vs Z's ~15M).
    """
    def __init__(self, in_features, out_features, generator, latent_dim, block_k, block_n):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.generator = generator
        self.num_k = math.ceil(in_features / block_k)
        self.num_n = math.ceil(out_features / block_n)
        self.padded_in = self.num_k * block_k
        self.padded_out = self.num_n * block_n
        self.block_k, self.block_n = block_k, block_n

        self.Z = nn.Parameter(torch.randn(self.num_k, self.num_n, latent_dim) * (0.23 / math.sqrt(latent_dim)))

        # Per-output-column log_scale. zeros() so exp(log_scale) starts at 1 for
        # every column, recovering scalar-log_scale behavior at init.
        self.log_scale = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('scale_init', torch.tensor(1.0 / math.sqrt(in_features)))

    def forward(self, x, tiles=None):
        if tiles is None:
            tiles = self.generator(self.Z)

        # Per-output-column RMS (reduce over input dims num_k, bk) — unit-norm columns of W.
        # Not detached: the function is exactly scale-invariant in tile-space, so the
        # optimizer correctly sees zero gradient in the tile-magnitude direction.
        rms = tiles.pow(2).mean(dim=(0, 2), keepdim=True).add(1e-12).sqrt()
        tiles = tiles / rms

        # scale: shape (out_features,). When passed to _assemble_weight, the
        # final (W * scale) broadcasts column-wise, since W is (in_f, out_f)
        # and scale's last dim matches out_f. So column j of W gets multiplied
        # by scale[j], giving per-column magnitude control.
        scale = self.scale_init * self.log_scale.exp()
        W = _assemble_weight(tiles, scale, self.in_features, self.out_features)
        return x @ W

# =============================================================================
# Chunked cross-entropy (unchanged — already opaque to compiler)
# =============================================================================

class ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.bfloat16)
    def forward(ctx, hidden, weight, targets, chunk_size=4096):
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size

        B, S, D = hidden.shape
        hidden_flat = hidden.reshape(-1, D)
        targets_flat = targets.reshape(-1)
        total_tokens = hidden_flat.shape[0]

        total_loss = 0.0
        with torch.no_grad():
            for start in range(0, total_tokens, chunk_size):
                end = min(start + chunk_size, total_tokens)
                logits = F.linear(hidden_flat[start:end], weight)
                loss = F.cross_entropy(logits, targets_flat[start:end], reduction='sum')
                total_loss += loss.item()

        return hidden.new_tensor(total_loss / total_tokens)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        B, S, D = hidden.shape
        hidden_flat = hidden.reshape(-1, D)
        targets_flat = targets.reshape(-1)
        total_tokens = hidden_flat.shape[0]

        grad_hidden_flat = torch.empty_like(hidden_flat)
        grad_weight = torch.zeros_like(weight)

        scale = grad_output / total_tokens

        for start in range(0, total_tokens, chunk_size):
            end = min(start + chunk_size, total_tokens)
            h_chunk = hidden_flat[start:end]
            t_chunk = targets_flat[start:end]

            with torch.enable_grad():
                logits = F.linear(h_chunk, weight)

            probs = F.softmax(logits.float(), dim=-1)
            probs.scatter_add_(1, t_chunk.unsqueeze(1), torch.full_like(probs[:, :1], -1.0))
            grad_logits = (probs * scale).to(weight.dtype)

            grad_weight.addmm_(grad_logits.t(), h_chunk)
            grad_hidden_flat[start:end] = grad_logits @ weight

        return grad_hidden_flat.view(B, S, D), grad_weight, None, None

def chunked_cross_entropy(hidden, lm_head_weight, targets, chunk_size=4096):
    return ChunkedCrossEntropy.apply(hidden, lm_head_weight, targets, chunk_size)

# =============================================================================
# Standard Transformer (unchanged — baseline reference)
# =============================================================================

class StandardAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.d_head = c.n_heads, c.n_kv_heads, c.d_head
        self.n_rep = c.n_heads // c.n_kv_heads
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * c.d_head, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * c.d_head, bias=False)
        self.o_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.rope = RotaryEmbedding(c.d_head, c.max_seq_len)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        cos, sin = self.rope(S)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

class StandardBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.ffn_norm = RMSNorm(c.d_model)
        self.attn = StandardAttention(c)
        self.gate = nn.Linear(c.d_model, c.d_ffn, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ffn, bias=False)
        self.down = nn.Linear(c.d_ffn, c.d_model, bias=False)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x

class StandardTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList([StandardBlock(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        x = self.norm(x)
        if targets is not None:
            loss = chunked_cross_entropy(x, self.lm_head.weight, targets)
            return None, loss
        logits = self.lm_head(x)
        return logits, None

# =============================================================================
# Hypernetwork Transformer
# =============================================================================

class HyperAttention(nn.Module):
    def __init__(self, c, gen):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.d_head = c.n_heads, c.n_kv_heads, c.d_head
        self.n_rep = c.n_heads // c.n_kv_heads
        L, bk, bn = c.latent_dim, c.block_k, c.block_n
        self.q_proj = HyperLinear(c.d_model, c.d_model, gen, L, bk, bn)
        self.k_proj = HyperLinear(c.d_model, c.n_kv_heads * c.d_head, gen, L, bk, bn)
        self.v_proj = HyperLinear(c.d_model, c.n_kv_heads * c.d_head, gen, L, bk, bn)
        self.o_proj = HyperLinear(c.d_model, c.d_model, gen, L, bk, bn)
        self.rope = RotaryEmbedding(c.d_head, c.max_seq_len)

    def forward(self, x, tiles_list=None):
        B, S, D = x.shape
        if tiles_list is not None:
            q = self.q_proj(x, tiles=tiles_list[0])
            k = self.k_proj(x, tiles=tiles_list[1])
            v = self.v_proj(x, tiles=tiles_list[2])
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(S)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o_input = out.transpose(1, 2).contiguous().view(B, S, D)
        if tiles_list is not None:
            return self.o_proj(o_input, tiles=tiles_list[3])
        return self.o_proj(o_input)

class HyperBlock(nn.Module):
    def __init__(self, c, gen):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model)
        self.ffn_norm = RMSNorm(c.d_model)
        self.attn = HyperAttention(c, gen)
        L, bk, bn = c.latent_dim, c.block_k, c.block_n
        self.gate = HyperLinear(c.d_model, c.d_ffn, gen, L, bk, bn)
        self.up = HyperLinear(c.d_model, c.d_ffn, gen, L, bk, bn)
        self.down = HyperLinear(c.d_ffn, c.d_model, gen, L, bk, bn)

        #init_gate = 1.0 / math.sqrt(2 * c.n_layers)
        init_gate = 1.0
        self.attn_gate = nn.Parameter(torch.tensor(init_gate))
        self.ffn_gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, x, all_tiles=None):
        normed_x = self.attn_norm(x)

        if all_tiles is not None:
            attn_out = self.attn(normed_x, tiles_list=all_tiles[:4])
        else:
            attn_out = self.attn(normed_x)
        x = x + self.attn_gate * attn_out

        h = self.ffn_norm(x)

        if all_tiles is not None:
            ffn_out = self.down(
                F.silu(self.gate(h, tiles=all_tiles[4])) * self.up(h, tiles=all_tiles[5]),
                tiles=all_tiles[6]
            )
        else:
            ffn_out = self.down(F.silu(self.gate(h)) * self.up(h))

        x = x + self.ffn_gate * ffn_out
        return x


def _generate_layer_tiles(generator, layer, latent_dim):
    """Generate all 7 tile sets for one layer in a single batched generator call.

    Kept from V2 — batching reduces kernel launches by 7x per layer.
    The concatenated Z size is fixed for a given config (all layers share
    the same projection dimensions), so this doesn't cause recompilation
    when batch size changes.
    """
    projs = [
        layer.attn.q_proj, layer.attn.k_proj,
        layer.attn.v_proj, layer.attn.o_proj,
        layer.gate, layer.up, layer.down,
    ]
    Z_parts = []
    split_sizes = []
    orig_shapes = []
    for proj in projs:
        flat = proj.Z.reshape(-1, latent_dim)
        Z_parts.append(flat)
        split_sizes.append(flat.shape[0])
        orig_shapes.append(proj.Z.shape[:-1])

    Z_cat = torch.cat(Z_parts, dim=0)
    all_flat = generator(Z_cat)
    tile_splits = torch.split(all_flat, split_sizes, dim=0)
    return [
        t.view(*shape, generator.block_k, generator.block_n)
        for t, shape in zip(tile_splits, orig_shapes)
    ]


# Standalone function for torch.compile — avoids tracing Module.__call__
def _compiled_layer_fn(layer, x, tiles):
    return layer.forward(x, all_tiles=tiles)


class HyperTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.generator = TileGenerator(c.latent_dim, c.generator_hidden, c.block_k, c.block_n, c.generator_layers)
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList([HyperBlock(c, self.generator) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.latent_dim = c.latent_dim
        self._compiled_layer_fns = None
        self._init()

    def _init(self):
        gen_params = set(id(p) for p in self.generator.parameters())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if any(id(p) in gen_params for p in m.parameters()):
                    continue
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def compile_model(self):
        """
        Compile generator MLP + each layer body for speed.

        KEY CHANGES vs V2:
          - Generator compiled with dynamic=False: the Z_cat shape is fixed
            for a given config (it's determined by projection dimensions, not
            batch size). Static shapes let Triton fully autotune once.

          - Layer bodies compiled with dynamic=True: the B*S dimension changes
            with batch size. dynamic=True tells Dynamo to generate shape-generic
            code instead of recompiling for each new B*S. This is why V2 broke
            with batch size changes — it used dynamic=True on the generator
            (unnecessary) but the layer bodies' matmul shapes varied.

            Because _assemble_weight is hidden via allow_in_graph, the only
            shape-varying ops the compiler sees are standard matmuls (x @ W)
            which Triton handles robustly with dynamic shapes.
        """
        if _is_main_process():
            print("  Compiling generator MLP (dynamic=False, fixed Z shapes)...")
        self.generator.net = torch.compile(self.generator.net, dynamic=False)

        if _is_main_process():
            print("  Compiling layer bodies (dynamic=True, batch-robust)...")
        self._compiled_layer_fns = []
        for i, layer in enumerate(self.layers):
            compiled_fn = torch.compile(_compiled_layer_fn, dynamic=True)
            self._compiled_layer_fns.append(compiled_fn)
        if _is_main_process():
            print(f"  Compiled {len(self.layers)} layers.")

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)

        for i, layer in enumerate(self.layers):
            # Tiles generated OUTSIDE checkpoint — compiled generator runs here.
            # During backward, checkpoint provides these tiles (saved as args)
            # and recomputes the layer body — the generator does NOT re-run.
            tiles = _generate_layer_tiles(self.generator, layer, self.latent_dim)

            if self._compiled_layer_fns is not None:
                x = torch.utils.checkpoint.checkpoint(
                    self._compiled_layer_fns[i], layer, x, tiles,
                    use_reentrant=False
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, tiles, use_reentrant=False
                )

        x = self.norm(x)
        if targets is not None:
            loss = chunked_cross_entropy(x, self.lm_head.weight, targets)
            return None, loss
        logits = self.lm_head(x)
        return logits, None

# =============================================================================
# Dataset — sharded token files from Google Drive
# =============================================================================

import random
import numpy as np


class TokenSource:
    """Memmap-backed token source. Each call to get_batch() samples
    random fixed-length windows uniformly from the underlying corpus.

    For multi-shard corpora, picks shard with probability proportional to
    size, then a random offset uniformly within that shard. This is the
    nanoGPT-style 'random offset on memmap' pattern: simple, decorrelated,
    and avoids the deterministic-chunk-boundary artifact of fixed-stride
    sampling.

    Memory: memmaps are recreated per get_batch() call to avoid the
    documented numpy memmap memory leak in long-running processes.
    The OS page cache makes repeated opens nearly free.
    """

    def __init__(self, data_dir, shard_names, seq_len, dtype=np.uint16):
        self.data_dir = Path(data_dir)
        self.shard_names = list(shard_names)
        self.seq_len = seq_len
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize

        self.shard_sizes = np.array(
            [(self.data_dir / n).stat().st_size // self.itemsize
             for n in self.shard_names],
            dtype=np.int64,
        )
        if self.shard_sizes.sum() == 0:
            raise ValueError(f"All shards empty: {self.shard_names}")
        self.shard_probs = self.shard_sizes / self.shard_sizes.sum()

    @property
    def total_tokens(self):
        return int(self.shard_sizes.sum())

    def get_batch(self, batch_size, rng):
        """Sample (input, target) pairs with random offsets.
        Args:
            batch_size: int
            rng: numpy.random.Generator
        Returns:
            (x, y): two pinned int64 CPU tensors of shape (batch_size, seq_len)
        """
        shard_idxs = rng.choice(
            len(self.shard_names), size=batch_size, p=self.shard_probs
        )
        # Open each unique shard once for this batch
        unique = np.unique(shard_idxs)
        mmaps = {
            int(s): np.memmap(self.data_dir / self.shard_names[int(s)],
                              dtype=self.dtype, mode='r')
            for s in unique
        }

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, sidx in enumerate(shard_idxs):
            data = mmaps[int(sidx)]
            max_off = len(data) - self.seq_len - 1
            if max_off < 0:
                raise RuntimeError(
                    f"Shard {self.shard_names[int(sidx)]} too small "
                    f"({len(data)} tokens) for seq_len={self.seq_len}"
                )
            off = int(rng.integers(0, max_off + 1))
            chunk = np.asarray(data[off : off + self.seq_len + 1], dtype=np.int64)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        return (torch.from_numpy(x).pin_memory(),
                torch.from_numpy(y).pin_memory())


def load_data(cfg):
    """Returns (train_src, val_src) — both TokenSource instances."""
    if cfg.dataset == "wikitext":
        return load_wikitext_from_hf(cfg)
    elif cfg.dataset == "fineweb":
        return load_fineweb_from_drive(cfg)
    else:
        raise ValueError(f"Unknown cfg.dataset: {cfg.dataset!r}. Use 'fineweb' or 'wikitext'.")


def load_fineweb_from_drive(cfg):
    """Load fineweb data prepared by tokenize_train.py + migrate_pt_to_bin.py.
    Expects meta.json with `format: "raw_bin"` and uint16 .bin shards."""
    data_dir = Path(cfg.data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json found at {meta_path}")
    meta = json.loads(meta_path.read_text())

    fmt = meta.get("format", "")
    if fmt != "raw_bin":
        raise RuntimeError(
            f"Expected meta.json format=='raw_bin', got {fmt!r}.\n"
            f"Run migrate_pt_to_bin.py --data-dir {data_dir} to convert "
            f"legacy .pt shards to .bin."
        )

    print(f"  Dataset: {meta.get('dataset', 'fineweb')} (format=raw_bin)")
    print(f"  Train tokens: {meta.get('train_tokens', '?'):,}")
    print(f"  Val tokens:   {meta.get('val_tokens', '?'):,}")
    print(f"  Train shards: {len(meta['train_shards'])}, "
          f"Val shards: {len(meta['val_shards'])}")

    train_src = TokenSource(data_dir, meta['train_shards'], cfg.max_seq_len)
    val_src   = TokenSource(data_dir, meta['val_shards'],   cfg.max_seq_len)
    return train_src, val_src


def load_wikitext_from_hf(cfg):
    """Tokenize wikitext-103 once, save as .bin, return TokenSources."""
    cache_dir = Path("./wikitext_data")
    train_bin = cache_dir / "train.bin"
    val_bin   = cache_dir / "val.bin"

    if not (train_bin.exists() and val_bin.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        from datasets import load_dataset
        print("  Downloading wikitext-103 from HuggingFace...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        import sentencepiece as spm
        sp_local = spm.SentencePieceProcessor()
        sp_local.Load("tokenizer.model")

        def tokenize_split(split_name):
            print(f"    Tokenizing {split_name}...")
            lines = [line for line in ds[split_name]["text"] if line.strip()]
            token_lists = sp_local.Encode(lines)
            tokens = [tok for sublist in token_lists for tok in sublist]
            arr = np.asarray(tokens, dtype=np.int64)
            assert arr.max() <= 65535, f"vocab too large for uint16: max={arr.max()}"
            return arr.astype(np.uint16)

        tokenize_split("train").tofile(train_bin)
        tokenize_split("validation").tofile(val_bin)

    print(f"  Dataset: wikitext-103 (cached at {cache_dir})")
    print(f"  Train tokens: {train_bin.stat().st_size // 2:,}")
    print(f"  Val tokens:   {val_bin.stat().st_size // 2:,}")
    train_src = TokenSource(cache_dir, ["train.bin"], cfg.max_seq_len)
    val_src   = TokenSource(cache_dir, ["val.bin"],   cfg.max_seq_len)
    return train_src, val_src

# =============================================================================
# LR schedules
# =============================================================================

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

def get_gen_lr(step, cfg):
    if cfg.gen_lr_flat:
        gen_lr = cfg.max_lr * cfg.gen_lr_scale
        if step < cfg.warmup_steps:
            return gen_lr * (step + 1) / cfg.warmup_steps
        return gen_lr
    else:
        return get_lr(step, cfg) * cfg.gen_lr_scale

def get_z_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.z_max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return cfg.z_min_lr + 0.5 * (cfg.z_max_lr - cfg.z_min_lr) * (1 + math.cos(math.pi * progress))

# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, step, cfg, name, results):
    """Save checkpoint. Always uses the unwrapped model's state_dict, so
    checkpoints saved from a DDP run can be loaded by single-process runs
    (and vice versa) without state-dict prefix mismatch."""
    raw_model = _unwrap_model(model)
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "results": results,
    }
    ckpt_path = ckpt_dir / f"std_step_{step}.pt"
    torch.save(ckpt, ckpt_path)
    if hasattr(raw_model, 'generator'):
        z_dict = {}
        for n, p in raw_model.named_parameters():
            if n.endswith('.Z'):
                z_dict[n] = p.data.cpu()
        z_path = ckpt_dir / f"z_latents_step_{step}.pt"
        torch.save(z_dict, z_path)
    print(f"  Checkpoint saved: {ckpt_path}")
    (Path(cfg.checkpoint_dir) / name / "results.json").write_text(json.dumps(results, indent=2, default=str))

def load_checkpoint(path, model, optimizer=None):
    raw_model = _unwrap_model(model)
    ckpt = torch.load(path, weights_only=False)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  Resumed from step {ckpt['step']}: {path}")
    return ckpt["step"], ckpt.get("results", {"steps": []})

# =============================================================================
# Training loop
# =============================================================================

def train_model(model, name, train_ds, val_ds, cfg, device, dtype,
                rank=0, local_rank=0, world_size=1, is_distributed=False):
    log = print if rank == 0 else (lambda *a, **k: None)

    log(f"\n{'='*72}")
    log(f"  TRAINING: {name}")
    log(f"{'='*72}")

    raw_model = _unwrap_model(model)
    unique_params = sum(p.numel() for p in set(raw_model.parameters()))
    log(f"  Unique params: {unique_params:,}")
    model.to(device).train()

    is_hyper = hasattr(raw_model, 'generator')
    seen = set()

    if is_hyper:
        gen_decay, gen_nodecay = [], []
        z_decay, z_nodecay = [], []
        other_decay, other_nodecay = [], []
        for n, p in raw_model.named_parameters():
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            # log_scale (per-output-column, 1-D shape (out_features,)) gets decay so the
            # exponential parameterization has a prior at 0 (i.e. exp(log_scale)=1).
            is_decay = (p.dim() >= 2 and 'emb' not in n) or n.endswith('.log_scale')
            if 'generator' in n:
                (gen_decay if is_decay else gen_nodecay).append(p)
            elif n.endswith('.Z'):
                (z_decay if is_decay else z_nodecay).append(p)
            else:
                (other_decay if is_decay else other_nodecay).append(p)

        gen_lr = cfg.max_lr * cfg.gen_lr_scale
        lr_mode = "constant after warmup" if cfg.gen_lr_flat else f"cosine × {cfg.gen_lr_scale}"
        log(f"  Generator LR: {gen_lr:.2e} ({lr_mode})")
        log(f"  Z min LR: {cfg.z_min_lr:.2e} (cosine {cfg.z_max_lr:.0e} → {cfg.z_min_lr:.0e})")
        log(f"  Generator params: {sum(p.numel() for p in gen_decay) + sum(p.numel() for p in gen_nodecay):,}")
        log(f"  Z params: {sum(p.numel() for p in z_decay) + sum(p.numel() for p in z_nodecay):,}")
        log(f"  Other (nodecay, RMSNorm & residual gates): {sum(p.numel() for p in other_nodecay):,}")

        optimizer = torch.optim.AdamW([
            {"params": other_decay, "lr": cfg.max_lr, "weight_decay": cfg.weight_decay, "_group": "other"},
            {"params": other_nodecay, "lr": cfg.max_lr, "weight_decay": 0.0, "_group": "other"},
            # Z-decay is 0: L2 on Z does not translate to L2 on W (rms cancels it) and
            # would just collapse the latent grid into its Jacobian-linear regime.
            {"params": z_decay, "lr": cfg.max_lr, "weight_decay": 0.0, "_group": "z"},
            {"params": z_nodecay, "lr": cfg.max_lr, "weight_decay": 0.0, "_group": "z"},
            # Generator MLP is the actual representational substrate — it needs decay.
            # Higher β2: generator gradient is averaged across all 112 projections,
            # so the signal is smoother than Standard layers and benefits from a
            # longer second-moment window.
            {"params": gen_decay, "lr": gen_lr, "weight_decay": cfg.weight_decay,
             "betas": (0.9, 0.98), "_group": "gen"},
            {"params": gen_nodecay, "lr": gen_lr, "weight_decay": 0.0,
             "betas": (0.9, 0.98), "_group": "gen"},
        ], betas=(0.9, 0.95), fused=True)
    else:
        decay_p, nodecay_p = [], []
        for n, p in raw_model.named_parameters():
            if not p.requires_grad or id(p) in seen: continue
            seen.add(id(p))
            (decay_p if p.dim() >= 2 and 'emb' not in n else nodecay_p).append(p)
        optimizer = torch.optim.AdamW([
            {"params": decay_p, "weight_decay": cfg.weight_decay},
            {"params": nodecay_p, "weight_decay": 0.0},
        ], lr=cfg.max_lr, betas=(0.9, 0.95), fused=True)

    scaler = torch.amp.GradScaler('cuda') if dtype == torch.float16 else None

    start_step = 0
    results = {"steps": []}
    if cfg.resume_from:
        start_step, results = load_checkpoint(cfg.resume_from, model, optimizer)

    # ---- Data: nanoGPT-style get_batch on a TokenSource ----
    # No DataLoader, no IterableDataset, no num_workers. The TokenSource
    # samples random offsets via numpy, which is fast on CPU and the
    # OS page cache makes memmap reads nearly free.
    train_src = train_ds  # train_ds is a TokenSource (named for symmetry)
    val_src = val_ds
    if not isinstance(train_src, TokenSource):
        raise TypeError(
            f"train_ds must be a TokenSource, got {type(train_src).__name__}. "
            f"Update load_data() output to return TokenSource instances."
        )

    # Separate streams for train vs val so train shuffling doesn't leak
    # into eval, which simplifies bisect-style debugging. Per-rank seeds
    # so DDP ranks read different data; val_rng is shared across ranks
    # so all ranks evaluate the same windows and the all_reduced loss
    # is unbiased.
    data_rng = np.random.default_rng(seed=1337 + rank)
    val_rng = np.random.default_rng(seed=2024)

    fwd_times, bwd_times, opt_times = [], [], []

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    total_tokens = 0
    start_time = time.time()
    running_loss, n_loss = 0.0, 0

    if is_hyper and start_step == 0 and rank == 0:
        with torch.no_grad():
            layer0 = raw_model.layers[0]
            projs = [
                ("q_proj", layer0.attn.q_proj),
                ("k_proj", layer0.attn.k_proj),
                ("gate",   layer0.gate),
                ("down",   layer0.down),
            ]
            tiles = {}
            for proj_name, proj in projs:
                t = raw_model.generator(proj.Z)
                tiles[proj_name] = t.reshape(-1)

            log(f"\n  === Tile Diversity at Init ===")
            names = list(tiles.keys())
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    a, b = tiles[names[i]], tiles[names[j]]
                    L = min(len(a), len(b))
                    cos_sim = F.cosine_similarity(a[:L].unsqueeze(0), b[:L].unsqueeze(0)).item()
                    diff_std = (a[:L] - b[:L]).std().item()
                    log(f"    {names[i]} vs {names[j]}: cosine_sim={cos_sim:.6f}, diff_std={diff_std:.6f}")

            q_tiles = raw_model.generator(layer0.attn.q_proj.Z)
            q_flat = q_tiles.reshape(q_tiles.shape[0] * q_tiles.shape[1], -1)
            tile_mean = q_flat.mean(dim=0)
            deviations = (q_flat - tile_mean).norm(dim=1)
            log(f"    q_proj tile mean norm: {tile_mean.norm().item():.4f}")
            log(f"    q_proj tile deviation from mean: {deviations.mean().item():.4f} ± {deviations.std().item():.4f}")
            log(f"    q_proj inter-tile cosine sim (first 10 pairs):")
            for k in range(min(10, len(q_flat)-1)):
                cs = F.cosine_similarity(q_flat[k].unsqueeze(0), q_flat[k+1].unsqueeze(0)).item()
                log(f"      tile[{k}] vs tile[{k+1}]: {cs:.6f}")

            log(f"\n  === Learnable Scales & Gates at Init ===")
            # log_scale is now per-column. Print mean and spread so we can see
            # both the central tendency and any per-column drift during training.
            def _scale_stats(linear):
                s = (linear.scale_init * linear.log_scale.exp()).float()
                return s.mean().item(), s.std().item()
            q_mean, q_std = _scale_stats(layer0.attn.q_proj)
            d_mean, d_std = _scale_stats(layer0.down)
            log(f"    layer0 q_proj.scale = {q_mean:.6f} ±{q_std:.4f} (target {1.0/math.sqrt(2048):.6f})")
            log(f"    layer0 down.scale   = {d_mean:.6f} ±{d_std:.4f} (target {1.0/math.sqrt(8192):.6f})")
            log(f"    layer0 attn_gate    = {layer0.attn_gate.item():.6f} (target {1.0/math.sqrt(2*cfg.n_layers):.6f})")
            log(f"    layer0 ffn_gate     = {layer0.ffn_gate.item():.6f} (target {1.0/math.sqrt(2*cfg.n_layers):.6f})")
            log()

    # Compile AFTER diagnostics, BEFORE DDP wrapping and training loop.
    # Compiling the inner model first (then wrapping in DDP) is the standard
    # ordering and works well with DDP's bucketed gradient all-reduce.
    if is_hyper and cfg.use_compile:
        raw_model.compile_model()

    # Wrap in DDP after compile so DDP sees the compiled forward.
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
        # raw_model continues to point at the underlying nn.Module so
        # diagnostic accesses to .layers, .generator etc. still work.

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        if is_hyper:
            z_lr = get_z_lr(step, cfg)
            gen_lr = get_gen_lr(step, cfg)
        else:
            z_lr, gen_lr = None, None

        for pg in optimizer.param_groups:
            group = pg.get("_group", None)
            if group == "gen": pg["lr"] = gen_lr
            elif group == "z": pg["lr"] = z_lr
            else: pg["lr"] = lr

        optimizer.zero_grad()

        s_fwd = torch.cuda.Event(enable_timing=True)
        e_fwd = torch.cuda.Event(enable_timing=True)
        s_bwd = torch.cuda.Event(enable_timing=True)
        e_bwd = torch.cuda.Event(enable_timing=True)
        s_opt = torch.cuda.Event(enable_timing=True)
        e_opt = torch.cuda.Event(enable_timing=True)

        fwd_ms, bwd_ms = 0.0, 0.0

        for _ in range(cfg.grad_accum_steps):
            x, y = train_src.get_batch(cfg.batch_size, data_rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # total_tokens counts tokens across the whole DDP group, since each
            # rank consumed cfg.batch_size * seq_len tokens this micro-step.
            total_tokens += x.numel() * world_size

            s_fwd.record()
            with torch.amp.autocast('cuda', dtype=dtype):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            e_fwd.record()

            s_bwd.record()
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            e_bwd.record()

            torch.cuda.synchronize()
            fwd_ms += s_fwd.elapsed_time(e_fwd)
            bwd_ms += s_bwd.elapsed_time(e_bwd)
            running_loss += loss.item()
            n_loss += 1 / cfg.grad_accum_steps

        s_opt.record()
        if scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        e_opt.record()
        torch.cuda.synchronize()
        opt_ms = s_opt.elapsed_time(e_opt)

        if step >= cfg.timing_warmup:
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            opt_times.append(opt_ms)

        if (step + 1) % cfg.log_interval == 0:
            # All-reduce running loss across DDP ranks for accurate display.
            # Each rank's running_loss reflects its own batches; averaging
            # across ranks gives the global mean train loss.
            if is_distributed:
                loss_t = torch.tensor([running_loss, n_loss], device=device, dtype=torch.float32)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                global_running = loss_t[0].item()
                global_n = loss_t[1].item()
                avg_loss = global_running / max(global_n, 1)
            else:
                avg_loss = running_loss / max(n_loss, 1)

            elapsed = time.time() - start_time
            tok_s = total_tokens / elapsed
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            lr_extra = f" z_lr {z_lr:.2e} | gen_lr {gen_lr:.2e} |" if z_lr is not None else ""
            log(f"  [{name}] step {step+1:>5}/{cfg.max_steps} | "
                f"loss {avg_loss:.4f} | lr {lr:.2e} |{lr_extra} gnorm {gn:.2f} | "
                f"{tok_s:,.0f} tok/s | mem {peak_mem:.1f}GB | "
                f"fwd {fwd_ms:.1f}ms bwd {bwd_ms:.1f}ms opt {opt_ms:.1f}ms")
            running_loss, n_loss = 0.0, 0

        if (step + 1) % cfg.eval_interval == 0:
            model.eval()
            val_loss_local = 0.0
            eval_batches_local = 0
            for _ in range(cfg.eval_steps):
                vx, vy = val_src.get_batch(cfg.batch_size, val_rng)
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
                    _, vl = model(vx, vy)
                # fp32 cast on the scalar avoids bf16 quantization artifacts
                # (small numbers of eval batches snap to a 1/64-spaced grid)
                val_loss_local += vl.float().item()
                eval_batches_local += 1

            # Aggregate val loss across ranks. With val_rng shared, each rank
            # evaluates the same windows, so per-rank losses should be ~identical
            # modulo bf16 noise. We average to be safe and to make the reported
            # number reflect the full evaluation effort.
            if is_distributed:
                t = torch.tensor([val_loss_local, eval_batches_local],
                                 device=device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                val_loss = (t[0] / t[1]).item()
            else:
                val_loss = val_loss_local / max(eval_batches_local, 1)

            model.train()
            results["steps"].append({"step": step+1, "val_loss": round(val_loss, 4)})
            log(f"  [{name}] >>> VAL loss {val_loss:.4f} @ step {step+1}")

            if is_hyper and rank == 0:
                with torch.no_grad():
                    raw = _unwrap_model(model)
                    l0 = raw.layers[0]
                    lL = raw.layers[-1]
                    def _ms(linear):
                        s = (linear.scale_init * linear.log_scale.exp()).float()
                        return s.mean().item(), s.std().item()
                    l0_q_m, l0_q_s = _ms(l0.attn.q_proj)
                    l0_d_m, l0_d_s = _ms(l0.down)
                    lL_q_m, lL_q_s = _ms(lL.attn.q_proj)
                    lL_d_m, lL_d_s = _ms(lL.down)
                    log(f"  [{name}] scales: l0.q={l0_q_m:.4f}±{l0_q_s:.3f} "
                        f"l0.down={l0_d_m:.4f}±{l0_d_s:.3f} "
                        f"lL.q={lL_q_m:.4f}±{lL_q_s:.3f} "
                        f"lL.down={lL_d_m:.4f}±{lL_d_s:.3f} | "
                        f"gates: l0.attn={l0.attn_gate.item():.3f} l0.ffn={l0.ffn_gate.item():.3f} "
                        f"lL.attn={lL.attn_gate.item():.3f} lL.ffn={lL.ffn_gate.item():.3f}")

        if (step + 1) % cfg.save_interval == 0 and rank == 0:
            save_checkpoint(model, optimizer, step + 1, cfg, name, results)
        # All ranks need to participate in any subsequent collective ops, so
        # synchronize here to avoid the saver-rank-only path drifting ahead.
        if is_distributed:
            dist.barrier()

    # Final eval (same all-reduce pattern as periodic eval)
    model.eval()
    val_loss_local = 0.0
    eval_batches_local = 0
    for _ in range(cfg.eval_steps):
        vx, vy = val_src.get_batch(cfg.batch_size, val_rng)
        vx = vx.to(device, non_blocking=True)
        vy = vy.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            _, vl = model(vx, vy)
        val_loss_local += vl.float().item()
        eval_batches_local += 1
    if is_distributed:
        t_final = torch.tensor([val_loss_local, eval_batches_local],
                               device=device, dtype=torch.float32)
        dist.all_reduce(t_final, op=dist.ReduceOp.SUM)
        val_loss = (t_final[0] / t_final[1]).item()
    else:
        val_loss = val_loss_local / max(eval_batches_local, 1)

    def stats(arr):
        if not arr: return {"mean": 0, "std": 0}
        t = torch.tensor(arr)
        return {"mean": round(t.mean().item(), 1), "std": round(t.std().item(), 1)}

    results.update({
        "model": name,
        "unique_params": unique_params,
        "final_val_loss": round(val_loss, 4),
        "total_time_sec": round(elapsed, 1),
        "avg_tokens_per_sec": round(total_tokens / elapsed),
        "peak_mem_gb": round(peak_mem, 2),
        "timing": {
            "forward_ms": stats(fwd_times),
            "backward_ms": stats(bwd_times),
            "optimizer_ms": stats(opt_times),
        },
    })

    t = results["timing"]
    log(f"\n  [{name}] DONE: val_loss={val_loss:.4f} | {elapsed:.0f}s | "
        f"{total_tokens/elapsed:,.0f} tok/s | peak {peak_mem:.1f}GB")
    log(f"  Timing (mean±std ms): fwd {t['forward_ms']['mean']}±{t['forward_ms']['std']}  "
        f"bwd {t['backward_ms']['mean']}±{t['backward_ms']['std']}  "
        f"opt {t['optimizer_ms']['mean']}±{t['optimizer_ms']['std']}")

    # Final checkpoint (rank 0 only)
    if rank == 0:
        save_checkpoint(model, optimizer, cfg.max_steps, cfg, name, results)
    if is_distributed:
        dist.barrier()

    del optimizer
    if scaler: del scaler
    model.cpu(); gc.collect(); torch.cuda.empty_cache()
    return results

# =============================================================================
# Main
# =============================================================================

def main():
      # Mount Drive if in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass

    # Set up distributed first; this picks the correct GPU for this process
    # and lets us gate prints / file I/O on rank 0 from here on.
    rank, local_rank, world_size, is_distributed = _setup_distributed()
    log = print if rank == 0 else (lambda *a, **k: None)

    # If we did the rank-0-only download above, other ranks should now wait
    # at the barrier so they can read the downloaded file safely.
    if is_distributed:
        dist.barrier()

    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer.model")
    log(f"Vocab size: {sp.GetPieceSize()}")

    cfg = Config()

    # Validate dimension alignment before building the model
    cfg.validate_alignment()
    log("  Dimension alignment: OK (all dims are multiples of block_k/block_n)")

    # detect_gpu uses the rank's GPU. verbose=True only on rank 0 to keep logs clean.
    dtype = detect_gpu(local_rank=local_rank, verbose=(rank == 0))
    device = torch.device(f"cuda:{local_rank}")

    log(f"\n--- Distributed ---")
    log(f"  world_size={world_size}, rank={rank}, local_rank={local_rank}, distributed={is_distributed}")

    log(f"\n--- Config ---")
    log(f"  {cfg.n_layers}L × {cfg.d_model}d × {cfg.n_heads}h (FFN {cfg.d_ffn}), vocab {cfg.vocab_size}")
    log(f"  Seq {cfg.max_seq_len}, Per-GPU batch {cfg.batch_size}×{cfg.grad_accum_steps} = {cfg.effective_batch}")
    log(f"  Global batch (per optimizer step): {cfg.effective_batch * world_size}")
    log(f"  Steps: {cfg.max_steps}, Hyper: latent={cfg.latent_dim} block={cfg.block_k}×{cfg.block_n}")
    log(f"  Gen LR: {'flat' if cfg.gen_lr_flat else 'coupled'} (scale={cfg.gen_lr_scale})")
    log(f"  Compile: {cfg.use_compile}")

    if cfg.dataset == "fineweb":
        log(f"  Data: fineweb @ {cfg.data_dir}")
    else:
        log(f"  Data: {cfg.dataset}")
    log(f"  Checkpoints: {cfg.checkpoint_dir}")


    train_ds, val_ds = load_data(cfg)

    all_results = {}

    if cfg.run_hyper:
        model = HyperTransformer(cfg)
        all_results["hyper"] = train_model(
            model, "hyper", train_ds, val_ds, cfg, device, dtype,
            rank=rank, local_rank=local_rank,
            world_size=world_size, is_distributed=is_distributed,
        )
        del model; gc.collect(); torch.cuda.empty_cache()

    if cfg.run_standard:
        model = StandardTransformer(cfg)
        all_results["standard"] = train_model(
            model, "standard", train_ds, val_ds, cfg, device, dtype,
            rank=rank, local_rank=local_rank,
            world_size=world_size, is_distributed=is_distributed,
        )
        del model; gc.collect(); torch.cuda.empty_cache()

    # --- Comparison (rank 0 only) ---
    if rank == 0:
        keys = list(all_results.keys())
        if len(keys) >= 2:
            log(f"\n{'='*80}")
            log(f"  COMPARISON")
            log(f"{'='*80}")

            col = 18
            header = f"{'Metric':<28}" + "".join(f" {k:>{col}}" for k in keys)
            log(f"\n{header}")
            log("-" * len(header))

            for label, fn in [
                ("Trainable params", lambda k: f"{all_results[k]['unique_params']:,}"),
                ("Final val loss", lambda k: f"{all_results[k]['final_val_loss']:.4f}"),
                ("Peak memory (GB)", lambda k: f"{all_results[k]['peak_mem_gb']:.2f}"),
                ("Tokens/sec", lambda k: f"{all_results[k]['avg_tokens_per_sec']:,}"),
                ("Forward (ms)", lambda k: f"{all_results[k]['timing']['forward_ms']['mean']}"),
                ("Backward (ms)", lambda k: f"{all_results[k]['timing']['backward_ms']['mean']}"),
                ("Optimizer (ms)", lambda k: f"{all_results[k]['timing']['optimizer_ms']['mean']}"),
            ]:
                row = f"{label:<28}" + "".join(f" {fn(k):>{col}}" for k in keys)
                log(row)

            log(f"\n--- Val Loss ---")
            header = f"{'Step':>8}" + "".join(f" {k:>14}" for k in keys)
            if "standard" in all_results:
                for k in keys:
                    if k != "standard": header += f" {'Δ'+k:>14}"
            log(header)

            step_losses = {k: {s["step"]: s["val_loss"] for s in all_results[k].get("steps", [])} for k in keys}
            all_steps = sorted(set(s for sl in step_losses.values() for s in sl))
            for step in all_steps:
                row = f"{step:>8}"
                for k in keys:
                    v = step_losses[k].get(step, float('nan'))
                    row += f" {v:>14.4f}"
                if "standard" in all_results:
                    sv = step_losses["standard"].get(step, float('nan'))
                    for k in keys:
                        if k != "standard":
                            hv = step_losses[k].get(step, float('nan'))
                            d = hv - sv if not (math.isnan(sv) or math.isnan(hv)) else float('nan')
                            row += f" {d:>+14.4f}"
                log(row)

        results_path = Path(cfg.checkpoint_dir) / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log(f"\nResults saved to {results_path}")

    # Clean up the process group cleanly so torchrun doesn't hang.
    _cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()