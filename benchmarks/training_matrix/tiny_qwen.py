"""A dependency-free, small Qwen-style causal language model.

It is intentionally modest enough for CPU smoke tests but retains the
optimizer-relevant topology: tied token embeddings/LM head, separate Q/K/V/O
hidden matrices, RMSNorm, RoPE, grouped-query attention, and a SwiGLU MLP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_QWEN_PRESETS = {
    # ~33M screening run: fast enough to reject bad recipes before the longer
    # pretraining matrix while retaining repeated K/V matrix shapes.
    "screen_33m": {
        "hidden_size": 512,
        "intermediate_size": 1365,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
    },
    # 134,216,576 parameters with the byte vocabulary and tied LM head.  The
    # 14:2 GQA split yields head_dim=64 and repeated 128x896 K/V matrices,
    # exercising the batched Newton-Schulz shapes on a 24 GiB RTX 3090.
    "pretrain_134m": {
        "hidden_size": 896,
        "intermediate_size": 2432,
        "num_hidden_layers": 16,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    },
}


@dataclass(frozen=True)
class TinyQwenConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    hidden_size: int = 192
    intermediate_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


def preset_config(
    name: str,
    *,
    max_seq_len: int,
    rope_theta: float = 1_000_000.0,
    tie_word_embeddings: bool = True,
) -> TinyQwenConfig:
    try:
        dimensions = TINY_QWEN_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown TinyQwen preset {name!r}; choose from {sorted(TINY_QWEN_PRESETS)}") from exc
    return TinyQwenConfig(
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        tie_word_embeddings=tie_word_embeddings,
        **dimensions,
    )


def expected_parameter_count(config: TinyQwenConfig) -> int:
    """Analytic count for the exact bias-free/tied architecture."""

    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim
    embedding = config.vocab_size * config.hidden_size
    attention = (
        2 * config.hidden_size * config.hidden_size
        + 2 * config.hidden_size * kv_width
        + 2 * head_dim
    )
    mlp = 3 * config.hidden_size * config.intermediate_size
    layer_norms = 2 * config.hidden_size
    return embedding + config.num_hidden_layers * (attention + mlp + layer_norms) + config.hidden_size


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        source_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(source_dtype)


def _apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply RoPE to ``[batch, heads, sequence, head_dim]``."""

    head_dim = x.shape[-1]
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class TinyQwenAttention(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        kv_width = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Qwen3 applies per-head Q/K normalization before RoPE.
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(self.q_norm(q), self.rope_theta)
        k = _apply_rope(self.k_norm(k), self.rope_theta)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o_proj(attended)


class TinyQwenMLP(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TinyQwenDecoderLayer(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TinyQwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TinyQwenMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyQwenBackbone(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyQwenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


@dataclass
class TinyCausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    loss_sum: torch.Tensor | None
    supervised_tokens: torch.Tensor | None


class TinyQwenForCausalLM(nn.Module):
    def __init__(self, config: TinyQwenConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.model = TinyQwenBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> TinyCausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len={self.config.max_seq_len}"
            )
        logits = self.lm_head(self.model(input_ids))
        if labels is None:
            return TinyCausalLMOutput(logits, None, None, None)
        if labels.shape != input_ids.shape:
            raise ValueError("labels and input_ids must have the same shape")
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        supervised = (shift_labels != -100).sum()
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss = loss_sum / supervised.clamp_min(1)
        return TinyCausalLMOutput(logits, loss, loss_sum, supervised)
