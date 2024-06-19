from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "cifar100"
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 64
CFG.DATASET.SCALE = (224,224)

# Distiller
CFG.DISTILLER = CN()
CFG.DISTILLER.TYPE = "NONE"  # Vanilla as default
CFG.DISTILLER.TEACHER = "ResNet50"
CFG.DISTILLER.STUDENT = "resnet32"
CFG.DISTILLER.TEACHER_PATH = "output/opensarship/vanilla,teacher,res50/student_best"

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 64
CFG.SOLVER.EPOCHS = 240
CFG.SOLVER.LR = 0.05
CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0001
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"

# Log
CFG.LOG = CN()
CFG.LOG.TENSORBOARD_FREQ = 500
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = False

# Distillation Methods

# KD CFG
CFG.KD = CN()
CFG.KD.TEMPERATURE = 4
CFG.KD.LOSS = CN()
CFG.KD.LOSS.CE_WEIGHT = 0.1
CFG.KD.LOSS.KD_WEIGHT = 0.9

# AT CFG
CFG.AT = CN()
CFG.AT.P = 2
CFG.AT.LOSS = CN()
CFG.AT.LOSS.CE_WEIGHT = 1.0
CFG.AT.LOSS.FEAT_WEIGHT = 1000.0
CFG.AT.LOSS.KD_WEIGHT = 9.0

# RKD CFG
CFG.RKD = CN()
CFG.RKD.DISTANCE_WEIGHT = 25
CFG.RKD.ANGLE_WEIGHT = 50
CFG.RKD.LOSS = CN()
CFG.RKD.LOSS.CE_WEIGHT = 1.0
CFG.RKD.LOSS.FEAT_WEIGHT = 1.0
CFG.RKD.PDIST = CN()
CFG.RKD.PDIST.EPSILON = 1e-12
CFG.RKD.PDIST.SQUARED = False

# FITNET CFG
CFG.FITNET = CN()
CFG.FITNET.HINT_LAYER = 2  # (0, 1, 2, 3, 4)
CFG.FITNET.INPUT_SIZE = (32, 32)
CFG.FITNET.LOSS = CN()
CFG.FITNET.LOSS.CE_WEIGHT = 1.0
CFG.FITNET.LOSS.FEAT_WEIGHT = 100.0

# KDSVD CFG
CFG.KDSVD = CN()
CFG.KDSVD.K = 1
CFG.KDSVD.LOSS = CN()
CFG.KDSVD.LOSS.CE_WEIGHT = 1.0
CFG.KDSVD.LOSS.FEAT_WEIGHT = 1.0

# OFD CFG
CFG.OFD = CN()
CFG.OFD.LOSS = CN()
CFG.OFD.LOSS.CE_WEIGHT = 1.0
CFG.OFD.LOSS.FEAT_WEIGHT = 0.001
CFG.OFD.CONNECTOR = CN()
CFG.OFD.CONNECTOR.KERNEL_SIZE = 1

# NST CFG
CFG.NST = CN()
CFG.NST.LOSS = CN()
CFG.NST.LOSS.CE_WEIGHT = 1.0
CFG.NST.LOSS.FEAT_WEIGHT = 50.0

# PKT CFG
CFG.PKT = CN()
CFG.PKT.LOSS = CN()
CFG.PKT.LOSS.CE_WEIGHT = 1.0
CFG.PKT.LOSS.FEAT_WEIGHT = 30000.0

# SP CFG
CFG.SP = CN()
CFG.SP.LOSS = CN()
CFG.SP.LOSS.CE_WEIGHT = 1.0
CFG.SP.LOSS.FEAT_WEIGHT = 3000.0

# VID CFG
CFG.VID = CN()
CFG.VID.LOSS = CN()
CFG.VID.LOSS.CE_WEIGHT = 1.0
CFG.VID.LOSS.FEAT_WEIGHT = 1.0
CFG.VID.EPS = 1e-5
CFG.VID.INIT_PRED_VAR = 5.0
CFG.VID.INPUT_SIZE = (32, 32)

# CRD CFG
CFG.CRD = CN()
CFG.CRD.MODE = "exact"  # ("exact", "relax")
CFG.CRD.FEAT = CN()
CFG.CRD.FEAT.DIM = 128
CFG.CRD.FEAT.STUDENT_DIM = 256
CFG.CRD.FEAT.TEACHER_DIM = 256
CFG.CRD.LOSS = CN()
CFG.CRD.LOSS.CE_WEIGHT = 1.0
CFG.CRD.LOSS.FEAT_WEIGHT = 0.8
CFG.CRD.NCE = CN()
CFG.CRD.NCE.K = 16384
CFG.CRD.NCE.MOMENTUM = 0.5
CFG.CRD.NCE.TEMPERATURE = 0.07

# ReviewKD CFG
CFG.REVIEWKD = CN()
CFG.REVIEWKD.CE_WEIGHT = 1.0
CFG.REVIEWKD.REVIEWKD_WEIGHT = 1.0
CFG.REVIEWKD.WARMUP_EPOCHS = 20
CFG.REVIEWKD.SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.OUT_SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.IN_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.OUT_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.MAX_MID_CHANNEL = 512
CFG.REVIEWKD.STU_PREACT = False

# DKD(Decoupled Knowledge Distillation) CFG
CFG.DKD = CN()
CFG.DKD.CE_WEIGHT = 1.0
CFG.DKD.ALPHA = 1.0
CFG.DKD.BETA = 8.0
CFG.DKD.T = 4.0
CFG.DKD.WARMUP = 20

# CTKD
CFG.CTKD =CN()
CFG.CTKD.T_START = 0.05
CFG.CTKD.T_END = 5.0
CFG.CTKD.DECAY_MAX = 1.0
CFG.CTKD.DECAY_MIN = 0.0
CFG.CTKD.DECAY_LOOPS = 240
CFG.CTKD.H0 = 5.

#FDL
CFG.FDL = CN()
CFG.FDL.PATCH_SIZE = 3
CFG.FDL.STRIDE = 1
CFG.FDL.NUM_PROJ = 256
CFG.FDL.PHASE_WEIGHT = 0.
CFG.FDL.INPUT_SIZE = (32, 32)
CFG.FDL.WARMUP = 10
CFG.FDL.LOSS = CN()
CFG.FDL.LOSS.CE_WEIGHT = 1.0
CFG.FDL.LOSS.FDL_WEIGHT = 0.1
# SIMKD CFG
CFG.SIMKD = CN()
CFG.SIMKD.INPUT_SIZE = (32, 32)
CFG.SIMKD.CE_WEIGHT = 1.0
CFG.SIMKD.FEAT_WEIGHT = 100.0
CFG.SIMKD.FACTOR = 2
#FDL
CFG.FDL = CN()
CFG.FDL.PATCH_SIZE = 5
CFG.FDL.STRIDE = 1
CFG.FDL.NUM_PROJ = 256
CFG.FDL.PHASE_WEIGHT = 1
CFG.FDL.INPUT_SIZE = (32, 32)
CFG.FDL.WARMUP = 20
CFG.FDL.LOSS = CN()
CFG.FDL.LOSS.CE_WEIGHT = 1.0
CFG.FDL.LOSS.FDL_WEIGHT = 0.1

# DIST
CFG.DIST = CN()
CFG.DIST.BETA = 1.0
CFG.DIST.GAMMA = 1.0
CFG.DIST.TAU = 4.0
CFG.DIST.DIST_LOSS_WEIGHT = 2.0
CFG.DIST.CE_LOSS_WEIGHT = 1.0

#KENDALL
CFG.KENDALL = CN()
CFG.KENDALL.CE_WEIGHT = 1.
CFG.KENDALL.PCC_WEIGHT = 32.
CFG.KENDALL.KCC_WEIGHT = 32.