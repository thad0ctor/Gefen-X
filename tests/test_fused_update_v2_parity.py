"""
Parity test for the occupancy-flexible v2 fused update kernel.

v2 decouples the CUDA grid from num_blocks (see automatic_gefen_fused_kernel.cu)
so large-period / period==1 params are no longer near-serial. The math is
unchanged: the per-block magnitude is an absmax (order independent) and phase 2
recomputes the updated momentum with the identical float ops, so p, the stored
indices, and the magnitude must be bit-identical to the legacy v1 kernel.

Also checks the period==1 fast path of the vmean kernel against a PyTorch
reference computed with the same op order.

Run: python tests/test_fused_update_v2_parity.py
"""
import torch

from gefen.kernels.automatic_gefen_fused import _load_extension
from gefen.kernels.automatic_vmean import automatic_vmean_update_cuda

DEV = "cuda"

# (numel, period) covering every occupancy regime: huge period (few blocks),
# period==1 (one block per element), and the well-occupied middle.
CASES = [
    (1024 * 2048, 1048576),  # num_blocks = 2
    (1024 * 1024, 524288),   # num_blocks = 2
    (1024 * 3072, 98304),    # num_blocks = 32
    (3072 * 1024, 512),      # num_blocks = 6144
    (3072 * 1024, 64),       # num_blocks = 49152, threads = 64
    (2048 * 1024, 1),        # period == 1
    (1024 * 1024, 1),        # period == 1
    (128, 1),
]


def fused_parity():
    mod = _load_extension()
    codebook = torch.sort(torch.randn(256, device=DEV)).values.float().contiguous()
    ok = True
    for i, (numel, period) in enumerate(CASES):
        g = torch.Generator(device=DEV).manual_seed(100 + i)
        nb = numel // period
        p1 = (torch.randn(numel, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02).contiguous()
        gv = (torch.randn(nb, period, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
        ms1 = torch.randint(0, 256, (numel,), device=DEV, dtype=torch.uint8)
        mm1 = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) * 1e-3).contiguous()
        ss = (torch.rand(nb, 1, device=DEV, dtype=torch.float32, generator=g) + 0.5).contiguous()
        p2, ms2, mm2 = p1.clone(), ms1.clone(), mm1.clone()

        mod.automatic_gefen_fused_update_cuda(p1, gv, ms1, mm1, ss, codebook, False, 0.9, 5e-5)
        mod.automatic_gefen_fused_update_v2_cuda(p2, gv, ms2, mm2, ss, codebook, False, 0.9, 5e-5)

        dp = (p1.float() - p2.float()).abs().max().item()
        dm = (mm1 - mm2).abs().max().item()
        dsign = (ms1.int() - ms2.int()).abs().max().item()
        good = dp == 0.0 and dm == 0.0 and dsign == 0
        ok = ok and good
        print(f"[fused] numel={numel} period={period} num_blocks={nb}: "
              f"dp={dp:.3e} dmag={dm:.3e} dsign={dsign} {'PASS' if good else 'FAIL'}")
    return ok


def vmean_period1_parity():
    ok = True
    for numel in (128, 1024, 1024 * 1024, 2048 * 1024):
        g = torch.Generator(device=DEV).manual_seed(7)
        gv = (torch.randn(numel, 1, device=DEV, dtype=torch.bfloat16, generator=g) * 1e-3).contiguous()
        vm = (torch.rand(numel, 1, device=DEV, dtype=torch.float32, generator=g) * 1e-3).contiguous()
        # same op order as the kernel: mean_square = g*g, beta2*vm + (1-beta2)*ms
        gf = gv.float()
        ref = 0.999 * vm + (1.0 - 0.999) * (gf * gf)
        automatic_vmean_update_cuda(vm, gv, 0.999)
        d = (vm - ref).abs().max().item()
        good = d <= 1e-6
        ok = ok and good
        print(f"[vmean p=1] numel={numel}: max|kernel-ref|={d:.3e} {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    a = fused_parity()
    b = vmean_period1_parity()
    print("\nV2 PARITY OVERALL:", "PASS" if (a and b) else "FAIL")
    raise SystemExit(0 if (a and b) else 1)
