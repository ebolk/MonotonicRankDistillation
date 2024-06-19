import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
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

class MAKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(MAKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        for param in self.teacher.parameters():
            param.requires_grad = False
        # self.layer = len(self.teacher.get_stage_channels()) + 1
        self.layer = 4

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        feature_student = feature_student['feats']
        logits_teacher , _ = self.teacher(image)
        # logits_stus = [logits_student]
        logits_stus = []
        for i in range(self.layer):
            logits_stu, _ = self.teacher(feature_student[i],i)
            logits_stus.append(logits_stu)
            
        # losses
        # loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + sum([F.cross_entropy(logits_stu,target) for logits_stu in logits_stus]))
        # loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student, target) + F.cross_entropy(logits_stus[-1],target))
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * (kd_loss(logits_student,logits_teacher,self.temperature) + kd_loss(logits_stus[-1],logits_teacher,self.temperature))
        # loss_kd = self.kd_loss_weight * kd_loss(logits_student,logits_teacher,self.temperature)
        
        # loss_kd = self.kd_loss_weight * sum(([kd_loss(
        #     logits_stu, logits_teacher, self.temperature
        # ) for logits_stu in logits_stus]))
        # loss_kd = self.kd_loss_weight * sum(([dkd_loss(
        #     logits_stu, logits_teacher, target,self.alpha,self.beta ,self.temperature
        # ) for logits_stu in logits_stus])) / (self.layer-1)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict