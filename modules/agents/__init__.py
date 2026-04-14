REGISTRY = {}

from .n_rnn_agent import NRNNAgent
from .n_rrn_agent import NRRNAgent
from.n_RZTXTransEncoder import NRZTXTAgent
from .n_rrn_transformer import NRRNAttenAgent
from .n_wmgrnn_agent import NWMGRNNAgent
from .n_wmgrrn_agent import NWMGRRNAgent
from .n_memGraph_agent import NGraphRRNAgent
from .n_rnnGCN_agent import NRNNCNAgent

REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["n_rrn"] = NRRNAgent
REGISTRY["n_rezero"] = NRZTXTAgent
REGISTRY["n_rrntrans"] = NRRNAttenAgent
REGISTRY["wmgn_rnn"] = NWMGRNNAgent
REGISTRY["wmgn_rrn"] = NWMGRRNAgent
REGISTRY["memgraph"] = NGraphRRNAgent
REGISTRY["rnngcn"] = NRNNCNAgent


