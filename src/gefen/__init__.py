from importlib.metadata import PackageNotFoundError, version

try:
    # Single-sourced from the installed distribution metadata (name: gefen-x).
    __version__ = version("gefen-x")
except PackageNotFoundError:
    # Importing straight from a source checkout that was never `pip install`ed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "Gefen",
    "GefenMuon",
    "GefenMuonHybrid",
    "GefenDCPState",
    "GefenSavePlanner",
    "split_params_for_muon",
    "validate_split",
    "kernels",
    "__version__",
]


def __getattr__(name):
    if name == "Gefen":
        from .gefen import Gefen

        return Gefen
    if name == "GefenMuon":
        from .gefen_muon import GefenMuon

        return GefenMuon
    if name == "GefenMuonHybrid":
        from .hybrid import GefenMuonHybrid

        return GefenMuonHybrid
    if name == "GefenDCPState":
        from .dcp import GefenDCPState

        return GefenDCPState
    if name == "GefenSavePlanner":
        from .dcp import GefenSavePlanner

        return GefenSavePlanner
    if name in ("split_params_for_muon", "validate_split"):
        from . import params

        return getattr(params, name)
    if name == "kernels":
        # NOT `from . import kernels`: its fromlist handling re-enters this
        # __getattr__ before the submodule import runs, recursing forever.
        import importlib

        return importlib.import_module(".kernels", __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
