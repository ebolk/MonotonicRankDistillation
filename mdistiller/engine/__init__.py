from .trainer import BaseTrainer, CRDTrainer,TKDTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "tkd": TKDTrainer,
}
