import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()

def at_loss(g_s, g_t, p,block):
    res_s = [*g_s[1:]]
    res_t = [*g_t[1:]]
    for i in range(len(g_s)-1):
        res_s.append(g_s[i+1] - block[i](g_s[i],i==0))
        res_t.append(g_t[i+1] - block[i](g_t[i],i==0))
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(res_s, res_t)])


class resAT(Distiller):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """

    def __init__(self, student, teacher, cfg):
        super(resAT, self).__init__(student, teacher)
        self.p = cfg.AT.P
        self.ce_loss_weight = cfg.AT.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.AT.LOSS.KD_WEIGHT
        self.feat_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT
        self.block = nn.ModuleList([Embedding(32,64),Embedding(64,128),Embedding(128,256)])

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher,feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * at_loss(
            feature_student["feats"][:], feature_teacher["feats"][:], self.p,self.block
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat + F.kl_div(F.log_softmax(logits_student, dim=1), F.softmax(logits_teacher, dim=1), reduction="batchmean") * self.kd_loss_weight,
        }
        return logits_student, losses_dict

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.block.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.block.parameters():
            num_p += p.numel()
        return num_p

class Embedding(nn.Module):
    def __init__(self, inplanes, planes):
        super(Embedding, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(2,stride=2)
        self.relu = nn.ReLU(inplace=True)
        # kaiming init
    def forward(self, x,preact):
        if not preact:
            x = self.avgpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        return x