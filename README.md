# [Gefen: Optimized Stochastic Optimizer](https://arxiv.org/pdf/2606.13894)

Gefen is a drop-in replacement for the AdamW optimizer for memory-efficient
training. It keeps the familiar AdamW training recipe while dramatically
reducing optimizer-state memory: an 8x reduction in AdamW memory footprint, or
about 6.5 GiB saved per billion parameters, while maintaining AdamW-level
performance. The reduced memory footprint lets you train larger models or use
larger batch sizes and, as a result, achieve higher training throughput.
All it takes is changing two lines of code: import Gefen and replace the AdamW
optimizer constructor.

## Installation

Install from PyPI:

```bash
pip install gefen
```

Or install from source:

```bash
git clone https://github.com/ndvbd/Gefen
cd Gefen
pip install -e .
```

On the first CUDA run, Gefen builds its fused CUDA kernels with PyTorch JIT and
`nvcc`. This can take a few minutes. Later runs reuse the cached build for the
same Python, PyTorch, CUDA version, and Gefen source checkout.

This keeps the source install lightweight, but it requires a CUDA toolkit and
host compiler compatible with your PyTorch installation. In the future, we plan
to make this smoother with prebuilt wheels for common PyTorch/CUDA combinations.

## Quick Start

```python
import torch
from gefen import Gefen

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(128, 10).to(device)

# optimizer = torch.optim.AdamW(
optimizer = Gefen(  # Replace AdamW with Gefen:
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)

inputs = torch.randn(32, 128, device=device)
targets = torch.randint(0, 10, (32,), device=device)

logits = model(inputs)
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)

print('Finished successfully.')
```

## Hugging Face Trainer

Until native `optim="gefen"` support is released in Transformers, pass Gefen to
the Trainer with `optimizer_cls_and_kwargs`:

```python
from gefen import Gefen
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=1e-3,
    weight_decay=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizer_cls_and_kwargs=(
        Gefen,
        {
            "lr": training_args.learning_rate,
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
            "fused": True,
        },
    ),
)
```

### Distributed Training

Gefen is fully compatible with standard distributed training setups, including PyTorch DDP, PyTorch FSDP (including FSDP2 with `fully_shard`), and all flavors of DeepSpeed ZeRO. Gefen can be used like any other PyTorch optimizer in these workflows, with either `fused=True` or `fused=False`.


### Extension: Gefen-Muon

`GefenMuon` adds a Muon-style pseudo-orthogonalization step on the first moment (skipping the second moment), then quantizes those first moments to 8-bit with Gefen's partitioning quantization.

> **Scope — read this first.** Like Muon, `GefenMuon` optimizes **2D hidden weight matrices only**. It has *no* code path for embeddings, the LM head, or 1D parameters (norms, biases) — feeding it those will fail or misbehave. For a full model, use [`GefenMuonHybrid`](#full-models-gefenmuonhybrid) below, which routes the non-Muon parameters to `Gefen` for you.
>
> **Pass `(name, param)` pairs, not bare tensors.** `GefenMuon` keys its 8-bit codebook cache on each parameter's name. If you strip the names (e.g. `[p for _, p in pairs]`) every parameter collapses to the name `"none"` and the cache is corrupted. Always pass the named pairs through:

```python
from gefen import GefenMuon

# muon_named_params: list of (name, param) for your 2D hidden weight matrices.
optimizer = GefenMuon(muon_named_params, lr=lr)
```

#### Full models: `GefenMuonHybrid`

`GefenMuonHybrid` is the drop-in for training a **whole model** with Muon. It routes 2D hidden weight matrices to `GefenMuon` and everything else (embeddings, LM head, norms, biases) to `Gefen`, behind a single `torch.optim.Optimizer` interface. Both sub-optimizers are 8-bit, so the whole optimizer-state footprint stays small — unlike a stock Muon+AdamW setup where the AdamW half is full precision. Scope: single-GPU / DDP (FSDP2 sharded params are out of scope).

```python
from gefen import GefenMuonHybrid

def split_params_for_muon(model):
    """Muon -> 2D hidden weight matrices; Gefen backup -> everything else."""
    muon, backup = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matrix = (
            p.ndim == 2
            and not any(k in name.lower() for k in ("embed", "wte", "lm_head"))
        )
        (muon if is_hidden_matrix else backup).append((name, p))
    return muon, backup

muon_named_params, backup_named_params = split_params_for_muon(model)

optimizer = GefenMuonHybrid(
    muon_named_params,
    backup_named_params,
    lr=lr,
    weight_decay=0.0,
    fused=True,
)
```

It supports `step()`, `zero_grad()`, `state_dict()`/`load_state_dict()`, and LR schedulers (e.g. `torch.optim.lr_scheduler.StepLR(optimizer, ...)`) like any optimizer. Because its constructor takes two parameter lists rather than a single iterable, build it yourself and hand it to the Hugging Face `Trainer` via `optimizers=` (not `optimizer_cls_and_kwargs`):

```python
optimizer = GefenMuonHybrid(*split_params_for_muon(model), lr=training_args.learning_rate)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  optimizers=(optimizer, None))  # (optimizer, lr_scheduler)
```

## Testimonials

Have you tried Gefen and want to report your impressions privately or publicly?
We would be happy to hear about your experience. With your permission, we can
credit you and mention your work here.


## Citation

If you found this library useful, please consider citing our work:

```bibtex
@article{benedek2026gefen,
  title={Gefen: Optimized Stochastic Optimizer},
  author={Benedek, Nadav and Koren, Tomer and Fried, Ohad},
  journal={arXiv preprint arXiv:2606.13894},
  year={2026}
}
```

