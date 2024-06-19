def normalize(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

def inter_intra_loss(logits_student, logits_teacher, function):
    return function(logits_student.transpose(0,1), logits_teacher.transpose(0,1))+ function(logits_student, logits_teacher)

def loss_similarity(logits_student, logits_teacher):
    return 1- F.cosine_similarity(normalize(logits_student), normalize(logits_teacher), dim=1).mean()

class MRKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(MRKD, self).__init__(student, teacher)
        self.temperature = 32.
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.ratio = 0.5
        self.beta = 0.5

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        loss_ce = 1. * F.cross_entropy(logits_student, target)
        similarity_loss = 16. * inter_intra_loss(logits_student,logits_teacher, loss_similarity)
        ken_loss = 32. *  inter_intra_loss(logits_student,logits_teacher, self.kendall_loss)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_similar": similarity_loss,
            "loss_kendall": ken_loss
        }
        return logits_student, losses_dict
    
    def kendall_loss(self,logits_student, logits_teacher,ratio = 1, beta = 0.5):
        logits_student = normalize(logits_student)
        logits_teacher = normalize(logits_teacher)
        c = int(logits_teacher.shape[-1] * ratio)
        logits_teacher , indices= logits_teacher.topk(c, dim = 1)
        logits_student = logits_student.gather(1, indices)
        pair_num = (c * (c-1)) // 2
        c_pair = random.sample(sorted(combinations(list(range(c)), 2)), pair_num)
        logits_teacher_prank = logits_teacher[:, c_pair].diff().squeeze()
        logits_student_prank = logits_student[:, c_pair].diff().squeeze(-1)
        score = logits_teacher_prank * logits_student_prank
        beta = self.beta
        score = 1 / (1 + (-score * beta).exp())
        return 1-(2 * score - 1).mean(dim=-1)
    







import torch
import torch.nn.functional as F
import random
from itertools import combinations
from ._base import Distiller

def normalize(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

def kendall_loss(logits_student, logits_teacher,ratio = 1, beta = 0.5):
    logits_student = normalize(logits_student)
    logits_teacher = normalize(logits_teacher)
    c = int(logits_teacher.shape[-1] * ratio)
    logits_teacher , indices= logits_teacher.topk(c, dim = 1)
    logits_student = logits_student.gather(1, indices)
    pair_num = (c * (c-1)) // 2
    c_pair = random.sample(sorted(combinations(list(range(c)), 2)), pair_num)
    logits_teacher_prank = logits_teacher[:, c_pair].diff().squeeze()
    logits_student_prank = logits_student[:, c_pair].diff().squeeze(-1)
    score = logits_teacher_prank * logits_student_prank
    score = 1 / (1 + (-score * beta).exp())
    score = 1-(2 * score - 1).mean(dim=-1)
    return score