import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import Global_T


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    # pred_teacher_part2 = F.softmax(
    #     logits_teacher / temperature - 100.0 * gt_mask, dim=1
    # )
    # log_pred_student_part2 = F.log_softmax(
    #     logits_student / temperature - 100.0 * gt_mask, dim=1
    # )
    
    pred_teacher_part2 = F.softmax(
        logits_teacher[other_mask].reshape(logits_teacher.shape[0],-1)/ temperature, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student[other_mask].reshape(logits_student.shape[0],-1) / temperature, dim=1
    )
    
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class CTKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(CTKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        self.T = Global_T()
        self.T_START = cfg.CTKD.T_START
        self.END = cfg.CTKD.T_END
        self.gradient_decay = CosineDecay(max_value=cfg.CTKD.DECAY_MAX, min_value=cfg.CTKD.DECAY_MIN, num_loops=cfg.CTKD.DECAY_LOOPS)
        
    def get_learnable_parameters(self):
        res = [v for k, v in self.student.named_parameters()]
        res += [v for k, v in self.T.named_parameters()]
        return res
    
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        temp = self.T(logits_student, logits_teacher,self.gradient_decay.get_value(kwargs["epoch"]))
        temp = self.T_START + self.END * torch.sigmoid(temp)
        temp = temp.cuda()
        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temp,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
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