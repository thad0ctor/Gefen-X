"""Regression guard for the src/gefen/ layout migration (PR #50).

Under the old flat layout the package lived at the repo root, so the repo root
on ``sys.path`` shadowed things: ``gefen.py`` shadowed the installed ``gefen``
package (``import gefen`` failed from a clone), and the top-level ``kernels/``
dir shadowed Hugging Face's ``kernels`` package (breaking
``pip install --no-build-isolation .`` in envs that have HF ``kernels``).

The migration moved everything under ``src/gefen/``, which structurally removes
both shadows. This test asserts that invariant so a future change can't silently
re-introduce a top-level module/dir at the repo root and bring the shadowing
back. It is env-independent (no imports of the package, no installed deps).
"""
from pathlib import Path

# tests/ lives at the repo root; its parent is the repo root regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_flat_layout_module_shadows_at_repo_root():
    # These module files lived at the repo root under the flat layout and
    # shadowed the installed ``gefen`` package when cwd was the repo root.
    shadow_modules = [
        "gefen.py",
        "gefen_muon.py",
        "hybrid.py",
        "params.py",
        "partitioning.py",
        "quantization.py",
        "__init__.py",
    ]
    offenders = [m for m in shadow_modules if (REPO_ROOT / m).exists()]
    assert not offenders, (
        "flat-layout module file(s) at the repo root would shadow the installed "
        "gefen package: {}. Keep package modules under src/gefen/.".format(offenders)
    )


def test_no_top_level_package_dir_shadows_external_packages():
    # A top-level ``kernels/`` shadowed Hugging Face's ``kernels`` package (the
    # --no-build-isolation break); ``tools/`` was similarly ambiguous.
    for name in ("kernels", "tools"):
        assert not (REPO_ROOT / name).is_dir(), (
            "top-level {0}/ at the repo root shadows external packages (e.g. HF "
            "`kernels`); keep it under src/gefen/{0}/.".format(name)
        )


def test_package_lives_under_src_gefen():
    assert (REPO_ROOT / "src" / "gefen" / "__init__.py").exists(), (
        "the gefen package must live under src/gefen/ (src-layout)."
    )
    assert (REPO_ROOT / "src" / "gefen" / "kernels").is_dir()
