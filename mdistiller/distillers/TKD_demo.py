import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import Global_T, Mlp_T


def temp_loss(logits_teacher,logits_student,temp,label):
    #soft target
    pred_teacher = F.softmax(logits_teacher / temp, dim=1)
    pred_student = F.softmax(logits_student / temp, dim=1)
    # 
    return 0.5*temp*((pred_teacher[label]-pred_student[label]))**2




def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = (F.kl_div(log_pred_student, pred_teacher, reduction="none")
               .sum(1)*temperature**2).mean()
    return loss_kd



class TKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(TKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.T = Mlp_T(dim_in=student.fc.in_features + teacher.fc.in_features)
        self.T.cuda()
        self.T_START = cfg.CTKD.T_START
        self.T_END = cfg.CTKD.T_END
        self.H0 = cfg.CTKD.H0
        # self.gradient_decay = CosineDecay(max_value=cfg.CTKD.DECAY_MAX, min_value=cfg.CTKD.DECAY_MIN, num_loops=cfg.CTKD.DECAY_LOOPS)
    
    def get_learnable_parameters(self):
        res = [v for k, v in self.student.named_parameters()]
        return res
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.T.parameters():
            num_p += p.numel()
        return num_p
    
    def forward_train(self, image, target, **kwargs):
        logits_student, feats_stu = self.student(image)
        with torch.no_grad():
            logits_teacher, feats_tea = self.teacher(image)

        temp = self.T(feats_stu['pooled_feat'].detach(), feats_tea['pooled_feat'].detach(),1)
        # temp = self.T(logits_student, logits_teacher,1)
        temp = self.T_START + self.T_END * temp
        temp_scalar = temp.detach()
        
        # # losses
        # loss_t = 0.1*temp.mean()*(temp_loss(logits_teacher,temp,self.H0).detach())
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, temp_scalar
        )
        loss_temp = temp_loss(logits_teacher.detach(),logits_student.detach(),temp,target.detach().cpu().numpy()).mean()
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd, 
            "loss_t": loss_temp
        }
        return logits_student, losses_dict

class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value