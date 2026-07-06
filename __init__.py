__all__ = [
    "Gefen",
    "GefenMuon",
    "GefenMuonHybrid",
    "split_params_for_muon",
    "validate_split",
    "kernels",
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
    if name in ("split_params_for_muon", "validate_split"):
        from . import params

        return getattr(params, name)
    if name == "kernels":
        # NOT `from . import kernels`: its fromlist handling re-enters this
        # __getattr__ before the submodule import runs, recursing forever.
        import importlib

        return importlib.import_module(".kernels", __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
