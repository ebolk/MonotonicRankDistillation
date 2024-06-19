import torch.nn as nn
import torch
from torch.nn import functional as F
from ._base import Distiller

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def Brownian_correlation(x, y, eps=1e-8):
    cov = distance_covariance(x, y)
    sd = torch.sqrt(distance_variance(x) * distance_variance(y) + eps)
    return 1 - (cov / sd).mean()

def double_center(x):
    x = x - x.mean(dim=2, keepdim=True) - x.mean(dim=1, keepdim=True) + x.mean(dim=(1,2), keepdim=True)
    return x

def distance_covariance(x, y):
    N = x.shape[1]
    distX = torch.abs((x.unsqueeze(1) - x.unsqueeze(2)))
    distY = torch.abs((y.unsqueeze(1) - y.unsqueeze(2)))
    centeredX = double_center(distX)
    centeredY = double_center(distY)
    calc = torch.sum(centeredX * centeredY , dim = (1,2))
    return torch.sqrt(calc / (N ** 2))

def distance_variance(x):
    return distance_covariance(x, x)


class DIST(Distiller):
    def __init__(self, student,teacher,cfg):
        super(DIST, self).__init__(student,teacher)
        self.beta = cfg.DIST.BETA
        self.gamma = cfg.DIST.GAMMA
        self.tau = cfg.DIST.TAU
        self.ce_loss_weight = cfg.DIST.CE_LOSS_WEIGHT
        self.dist_loss_weight = cfg.DIST.DIST_LOSS_WEIGHT

    def dist_loss(self, z_s, z_t):# -> Any:
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss
    def distance_loss(self, z_s, z_t):
        return 32. * Brownian_correlation(z_s, z_t)
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        # losses
        loss_ce = F.cross_entropy(logits_student, target) * self.ce_loss_weight
        loss_dist = self.dist_loss(logits_student, logits_teacher) * self.dist_loss_weight
        # loss_distance = self.distance_loss(logits_student, logits_teacher) * self.dist_loss_weight
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_dist": loss_dist,
        }
        return logits_student, losses_dict
        
