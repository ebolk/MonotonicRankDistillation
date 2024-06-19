import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from ._base import Distiller

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, student):
        student = self.Connectors(student)
        return student



def statm_loss(x, y):
    x = x.view(x.size(0),x.size(1),-1)
    y = y.view(y.size(0),y.size(1),-1)
    x_mean = x.mean(dim=2)
    y_mean = y.mean(dim=2)
    mean_gap = (x_mean-y_mean).pow(2).mean(1)
    return mean_gap.mean()
    
class SRRL(Distiller):
    def __init__(self, student, teacher,cfg):
        super().__init__(student, teacher)
        self.connectors = transfer_conv(student.fc.in_features, teacher.fc.in_features)
        
    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.connectors.parameters())
    
    def get_extra_parameters(self):
        num_p = 0
        for p in self.connectors.parameters():
            num_p += p.numel()
        return num_p
    
    def forward_train(self,image,target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        f_s = feature_student["feats"][-1]
        f_t = feature_teacher["feats"][-1]
        f_s = self.connectors(f_s)
        
        loss_ce = F.cross_entropy(logits_student, target)
        loss_statm = statm_loss(f_s, f_t)
        
        pred_s = self.teacher.fc(self.teacher.avgpool(f_s).view(f_s.size(0), -1))
        
        loss_sr = 1. * F.mse_loss(pred_s,logits_teacher)
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_statm": loss_statm,
            "loss_sr": loss_sr,
        }
        
        return logits_student, losses_dict
    
        
