import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ._base import Distiller
# PPA
from .ppa import PPA
# dkd
from .DKD import DKD
# simkd
from .SimKD import SimKD
# BDC_FEAT
from .BDC_FEAT import BDCKDFEAT
# CVPR2024


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

# 


class BDC(nn.Module):

    def __init__(self, is_vec=True, input_dim=640, dimension_reduction=None, activate='relu'):
        super(BDC, self).__init__()
        self.is_vec = is_vec
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = input_dim[0]
        if self.dr is not None and self.dr != self.input_dim:
            if activate == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activate == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1)
            else:
                self.act = nn.ReLU(inplace=True)

            self.conv_dr_block = nn.Sequential(
                nn.Conv2d(self.input_dim, self.dr,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.dr),
                self.act
            )
        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim*(output_dim+1)/2)
        else:
            self.output_dim = int(output_dim*output_dim)

        # self.temperature = nn.Parameter(torch.log(
        #     (1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1, 1)), requires_grad=True)
        self.temperature = torch.log(
            (1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1, 1))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.dr is not None and self.dr != self.input_dim:
            x = self.conv_dr_block(x)
        x = BDCovpool(x, self.temperature)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = x.reshape(x.shape[0], -1)
        return x


def BDCovpool(x, t):
    batchSize, dim, M = x.data.shape
    # M = h * w
    # x = x.reshape(batchSize, dim, M)
    I = torch.eye(dim, dim, device=x.device).view(
        1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2

    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t).to(dcov.device) * dcov
    dcov = torch.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

    return t


def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple=False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2),
                    device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y


class BDCKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(BDCKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.BDC.CE_WEIGHT
        self.bdc_loss_weight = cfg.BDC.BDC_WEIGHT
        self.kd_loss_weight = cfg.BDC.KD_WEIGHT
        self.T = cfg.BDC.T
        # DKD
        if cfg.BDC.ADDDKD:
            self.dkd = DKD(student, teacher, cfg)
            self.dkd_loss_weight = cfg.BDC.DKD_WEIGHT
        # SimKD
        if cfg.BDC.ADDSIMKD:
            self.simkd = SimKD(student, teacher, cfg)
            self.simkd_loss_weight = cfg.BDC.SIMKD_WEIGHT
        if cfg.BDC.ADD_BDC_FEAT:
            # BDC_FEAT
            self.bdc_feat = BDCKDFEAT(student, teacher, cfg)
            self.bdc_feat_loss_weight = cfg.BDC.BDC_FEAT_WEIGHT

        self.is_logit_standardization = cfg.LOGIT_STANDARDIZATION
        self.add_attention = cfg.BDC.ADD_ATTENTION

        # # bdc
        # self.bdc_loss_weight = nn.Parameter(
        #     torch.tensor(self.bdc_loss_weight), requires_grad=True).cpu()
        # # kd loss1-bdc_loss_weight，
        # self.kd_loss_weight = 1 - self.bdc_loss_weight

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # SimKD，SimKDloss
        if hasattr(self, 'simkd'):
            logits_student, loss_simkd = self.simkd.forward_train(
                image, target, **kwargs)
            loss_simkd = loss_simkd["loss_kd"]
            loss_kd = loss_simkd * self.simkd_loss_weight
        else:
            # loss_kd
            loss_kd = 0

        logits_student = normalize(
            logits_student) if self.is_logit_standardization else logits_student
        logits_teacher = normalize(
            logits_teacher) if self.is_logit_standardization else logits_teacher

        # losses ce
        loss_ce = F.cross_entropy(logits_student, target)
        loss_ce = self.ce_loss_weight * loss_ce

        # losses kd
        p_s = F.log_softmax(logits_student/self.T, dim=1)
        p_t = F.softmax(logits_teacher/self.T, dim=1)
        loss_kl = F.kl_div(p_s, p_t, size_average=False) * \
            (self.T**2) / logits_student.shape[0]
        loss_kl = self.kd_loss_weight * loss_kl
        loss_kd = loss_kl + loss_kd

        # 
        y_s = logits_student.unsqueeze(2)
        y_t = logits_teacher.unsqueeze(2)

        # BDC
        bdc = BDC(is_vec=False, input_dim=y_s.shape, dimension_reduction=None)
        bdc_y_s = bdc(y_s)
        bdc_y_t = bdc(y_t)
        # bdc_y_sbdc_y_t
        bdc_y_s = F.normalize(bdc_y_s, p=2, dim=1)
        bdc_y_t = F.normalize(bdc_y_t, p=2, dim=1)
        # 
        # ，loss
        loss_bdc = - torch.mul(bdc_y_s, bdc_y_t).sum() * self.bdc_loss_weight

        # DKD，DKDloss
        if hasattr(self, 'dkd'):
            loss_dkd = self.dkd.forward_train(
                image, target, **kwargs)[1]["loss_kd"]
            loss_kd = loss_kd + loss_dkd * self.dkd_loss_weight

        if hasattr(self, 'bdc_feat'):
            loss_bdc_feat = self.bdc_feat.forward_train(
                image, target, **kwargs)[1]["loss_bdc"]
            loss_bdc_feat = loss_bdc_feat * self.bdc_feat_loss_weight
            loss_bdc = loss_bdc + loss_bdc_feat

        losses_dict = {
            "loss_ce": loss_ce.cuda(),
            "loss_bdc": loss_bdc.cuda(),
            "loss_kd": loss_kd.cuda(),
        }

        return logits_student, losses_dict

    def forward_test(self, image):
        if hasattr(self, 'simkd'):
            return self.simkd.forward_test(image)
        else:
            return self.student(image)[0]

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        # 
        return [v for k, v in self.student.named_parameters()] + ([v for k, v in self.simkd.simkd.named_parameters()] if hasattr(self, 'simkd') else [])
