from .rnn_agent import RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .ss_rnn_agent import SS_RNNAgent  

REGISTRY = {}


REGISTRY["rnn"] = RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["ss_rnn"] = SS_RNNAgent