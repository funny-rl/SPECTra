from .basic_controller import BasicMAC
from .hpn_controller import HPNMAC
from .ss_controller import SSMAC

REGISTRY = {}


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["hpn_mac"] = HPNMAC
REGISTRY["ss_mac"] = SSMAC