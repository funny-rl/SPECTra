from .CDS_QMIX import CDS_QMIX
from .CDS_QPLEX import CDS_QPLEX
from .nq_learner import NQLearner

REGISTRY = {}

REGISTRY["CDS_QMIX"] = CDS_QMIX
REGISTRY["CDS_QPLEX"] = CDS_QPLEX
REGISTRY["nq_learner"] = NQLearner
