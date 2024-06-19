import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def adaptivedkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_teacher, target)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pre_student = F.softmax(logits_student / temperature, dim=1)
    pt = (pred_teacher * gt_mask).sum(1)
    ps = (pre_student * gt_mask).sum(1)
    tckd_loss = (
       (1+pt)**alpha  * (- torch.log(ps))
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        (F.kl_div(log_pred_student_part2,pred_teacher_part2,reduction='none')).sum(dim=1) * temperature**2
    )
    return (tckd_loss + beta * nckd_loss).mean()


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


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss

class adaptiveDKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(adaptiveDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        # self.attention = 

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_adaptivedkd = min(kwargs["epoch"] / self.warmup, 1.0) * adaptivedkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_cd": loss_ce,
            "loss_kd": loss_adaptivedkd,
        }
        return logits_student, losses_dict
