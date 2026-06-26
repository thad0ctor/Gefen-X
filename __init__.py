__all__ = ["Gefen", "GefenMuon", "GefenMuonHybrid", "kernels"]


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
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
