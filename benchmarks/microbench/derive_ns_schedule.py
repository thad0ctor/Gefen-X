"""Derive tuned per-iteration Newton-Schulz coefficient schedules.

Each NS iteration acts on the singular values of the (Frobenius-normalized)
matrix as the odd quintic  p(s) = a s + b s^3 + c s^5 . Running K iterations
composes K such polynomials. Muon's orthogonalization wants every nonzero
singular value driven to 1, i.e. the K-fold composition f_K(s) ~ 1 over the
range of singular values that actually occurs.

This script:
  1. measures the empirical Frobenius-normalized singular-value distribution of
     the real Muon weight shapes (NS is scale-invariant -- it divides by the
     Frobenius norm -- so only the SHAPE matters, not the gradient magnitude);
  2. fits, for K in {3,4}, the 3K coefficients that minimize the Frobenius
     orthogonality residual  sum_i (f_K(s_i)^2 - 1)^2  weighted by that
     empirical distribution (this is exactly ||O^T O - I||_F^2 in sv space);
  3. reports the achieved residual vs the standard fixed 5-step quintic.

No GPU required (small SVDs + scalar optimization). Run with the project venv.
"""
import numpy as np
from scipy.optimize import least_squares

try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

STD_QUINTIC = (3.4445, -4.7750, 2.0315)

# Real Muon weight shapes (hidden=2048, inter=6144).
SHAPES = [
    (2048, 2048),   # q/k/v/o proj (square)
    (512, 2048),    # grouped k/v proj (wide)
    (6144, 2048),   # gate/up proj (tall)
    (2048, 6144),   # down proj (wide)
]


def singular_values_normalized(rows, cols, seed):
    """Frobenius-normalized singular values of a random Gaussian matrix.

    Uses GPU SVD when available (CPU float64 SVD of 6144x2048 is very slow)."""
    if _HAVE_TORCH and torch.cuda.is_available():
        g = torch.Generator(device="cuda").manual_seed(seed)
        A = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
        s = torch.linalg.svdvals(A).double()
        s = s / s.norm()
        return s.cpu().numpy()
    g = np.random.default_rng(seed)
    A = g.standard_normal((rows, cols)).astype(np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    s = s / np.sqrt((s ** 2).sum())  # divide by Frobenius norm
    return s


def collect_distribution(seeds=(0, 1, 2, 3), cache="/tmp/ns_svdist.npy"):
    import os
    if os.path.exists(cache):
        return np.load(cache)
    vals = []
    for (r, c) in SHAPES:
        for sd in seeds:
            vals.append(singular_values_normalized(r, c, sd))
    s = np.concatenate(vals)
    np.save(cache, s)
    return s


def compose(coeffs, s):
    """K-fold composition of p(x)=a x + b x^3 + c x^5 over flat coeff vector."""
    x = s
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(0, len(coeffs), 3):
            a, b, c = coeffs[i], coeffs[i + 1], coeffs[i + 2]
            x2 = x * x
            x = x * (a + x2 * (b + c * x2))
        # divergent restarts overflow to inf; map to a large finite penalty so
        # least_squares discards them instead of crashing on nan.
        x = np.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6)
    return x


def ortho_residual(coeffs, s, w):
    """sqrt(w)*(f_K(s)^2 - 1): least_squares minimizes the weighted Frobenius
    orthogonality error sum w*(sigma^2-1)^2."""
    f = compose(coeffs, s)
    return np.sqrt(w) * (f * f - 1.0)


def frob_err(coeffs, s, w):
    f = compose(coeffs, s)
    return float(np.sqrt(np.sum(w * (f * f - 1.0) ** 2)))


def robust_residual(coeffs, s_min, n_target=400, n_safe=400, overshoot_cap=1.30):
    """Robust minimax objective (Polar-Express style).

    Drives f_K(s) -> 1 uniformly-in-log over the FULL reachable range
    [s_min, 1] (NOT just the tiny-singular-value bulk), and adds a one-sided
    penalty forbidding f from exceeding ``overshoot_cap`` anywhere in [0, 1].
    The full-range target + overshoot cap is what makes the schedule SAFE on
    near-low-rank gradients (whose normalized top singular value -> 1), unlike a
    schedule overfit to the small-s Gaussian bulk."""
    target_grid = np.logspace(np.log10(s_min), 0.0, n_target)
    f = compose(coeffs, target_grid)
    r_target = f - 1.0
    safe_grid = np.linspace(1e-4, 1.0, n_safe)
    fs = compose(coeffs, safe_grid)
    over = np.maximum(np.abs(fs) - overshoot_cap, 0.0)
    return np.concatenate([r_target, 8.0 * over])


def reach_and_overshoot(coeffs, s_min):
    g = np.logspace(np.log10(s_min), 0.0, 600)
    f = compose(coeffs, g)
    max_dev = float(np.max(np.abs(f - 1.0)))  # worst |f-1| on [s_min,1]
    g2 = np.linspace(1e-4, 1.0, 600)
    max_f = float(np.max(np.abs(compose(coeffs, g2))))  # peak overshoot on [0,1]
    return max_dev, max_f


def fit_robust(K, s_min, n_restarts=80, seed=0):
    """Fit a robust K-step schedule targeting full range [s_min, 1]."""
    rng = np.random.default_rng(seed)
    best, best_obj = None, np.inf
    base_inits = [
        np.tile(STD_QUINTIC, K),
        np.concatenate([[8.0, -23.0, 17.0]] + [list(STD_QUINTIC)] * (K - 1)),
    ]
    for r in range(n_restarts):
        if r < len(base_inits):
            x0 = np.array(base_inits[r], dtype=np.float64)
        else:
            a0 = np.linspace(rng.uniform(4, 8.5), rng.uniform(1.7, 2.6), K)
            x0 = np.empty(3 * K)
            for k in range(K):
                x0[3 * k] = a0[k]
                x0[3 * k + 1] = -(a0[k] * rng.uniform(1.0, 1.7))
                x0[3 * k + 2] = a0[k] * rng.uniform(0.3, 0.65)
        try:
            res = least_squares(
                robust_residual, x0, args=(s_min,), method="lm", max_nfev=20000
            )
        except Exception:
            continue
        obj = float(np.sum(res.x.size and robust_residual(res.x, s_min) ** 2))
        max_dev, max_f = reach_and_overshoot(res.x, s_min)
        # reject schedules that overshoot badly (unsafe on near-low-rank inputs)
        if max_f > 1.45:
            continue
        if obj < best_obj:
            best_obj, best = obj, res.x
    return best


def pick_smin(K, s, w, candidates):
    """For each candidate s_min, fit a robust schedule and score it by the
    ACTUAL Frobenius orthogonality residual on the empirical real distribution.
    Pick the s_min whose robust schedule minimizes real-data residual while
    staying safe."""
    best = None
    for s_min in candidates:
        coeffs = fit_robust(K, s_min)
        if coeffs is None:
            continue
        err = frob_err(coeffs, s, w)
        max_dev, max_f = reach_and_overshoot(coeffs, s_min)
        print(f"    s_min={s_min:.1e}: real-Frob={err:.4e}  "
              f"max|f-1|_[s_min,1]={max_dev:.3f}  peak|f|_[0,1]={max_f:.3f}")
        if best is None or err < best[1]:
            best = (coeffs, err, s_min, max_dev, max_f)
    return best


def main():
    s = collect_distribution()
    print(f"pooled singular values: {s.size} samples")
    print(f"  s range: [{s.min():.3e}, {s.max():.3e}]  median={np.median(s):.3e}")
    # weight by empirical density via histogram on log scale so all decades count
    s = np.clip(s, 1e-8, None)
    # equal weight per sample (empirical distribution); add a small log-grid
    # floor so very small (rare) sv's are not ignored.
    floor = np.logspace(-4, 0, 60)
    s_all = np.concatenate([s, floor])
    w = np.concatenate([np.ones_like(s), 0.05 * np.ones_like(floor)])
    w = w / w.sum()

    std5 = np.tile(STD_QUINTIC, 5)
    print(f"\nstandard 5-step quintic   Frob residual = {frob_err(std5, s_all, w):.4e}")
    std3 = np.tile(STD_QUINTIC, 3)
    std4 = np.tile(STD_QUINTIC, 4)
    print(f"standard coeffs @3 steps  Frob residual = {frob_err(std3, s_all, w):.4e}")
    print(f"standard coeffs @4 steps  Frob residual = {frob_err(std4, s_all, w):.4e}")

    smin_grid = {
        3: [3e-2, 2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3],
        4: [1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3],
    }
    for K in (3, 4):
        print(f"\n=== fitting robust {K}-step schedules (sweep s_min) ===")
        coeffs, err, s_min, max_dev, max_f = pick_smin(K, s_all, w, smin_grid[K])
        print(f"  -> chosen s_min={s_min:.1e}  real-Frob={err:.4e}  "
              f"peak|f|_[0,1]={max_f:.3f}")
        sched = []
        for k in range(K):
            a, b, c = coeffs[3 * k], coeffs[3 * k + 1], coeffs[3 * k + 2]
            sched.append((round(a, 4), round(b, 4), round(c, 4)))
            print(f"  ({a:+.4f}, {b:+.4f}, {c:+.4f}),")
        rounded = np.array(sched, dtype=np.float64).reshape(-1)
        rd, rf = reach_and_overshoot(rounded, s_min)
        print(f"  rounded real-Frob = {frob_err(rounded, s_all, w):.4e}  "
              f"peak|f|_[0,1]={rf:.3f}")


if __name__ == "__main__":
    main()
