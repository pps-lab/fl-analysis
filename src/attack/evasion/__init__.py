
from .norm import NormBoundPGDEvasion
from .trimmed_mean import TrimmedMeanEvasion
from .norm_prob_check import NormBoundProbabilisticCheckingEvasion
from .neurotoxin import NeurotoxinEvasion

def construct_evasion(classname, **kwargs):
    """Constructs evasion method"""
    import src.attack.evasion as ev
    cls = getattr(ev, classname)
    return cls(**kwargs)
