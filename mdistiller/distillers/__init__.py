from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .adaptiveDKD import adaptiveDKD
from .CTKD import CTKD
from .TKD_demo import TKD
from .resAT import resAT
from .FDL import FDL
from .MAKD import MAKD
from .SimKD import  SimKD
from .DIST import DIST
from .SRRL import SRRL
from .MLKD import MLKD
from .MRKD import MRKD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "adaptiveDKD": adaptiveDKD,
    "CTKD": CTKD,
    "TKD":TKD,
    "resAT": resAT,
    "FDL":FDL,
    "MAKD": MAKD,
    "SimKD": SimKD,
    "DIST":DIST,
    "SRRL":SRRL,
    "MLKD":MLKD,
    "MRKD":MRKD
}
