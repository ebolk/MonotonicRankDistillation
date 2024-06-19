import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import get_feat_shapes

class SimKD_module(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""
    def __init__(self, *, s_n, t_n, factor=2): 
        super(SimKD_module, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))       

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        
        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            # depthwise convolution
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))
      
    def forward(self, feat_s, feat_t, cls_t):
        
        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        
        trans_feat_t=target
        
        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)
        
        return trans_feat_s, trans_feat_t, pred_feat_s
# TODO: test
class SimKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(SimKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.SIMKD.CE_WEIGHT
        self.feat_loss_weight = cfg.SIMKD.FEAT_WEIGHT
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.SIMKD.INPUT_SIZE
        )
        s_n = feat_s_shapes[-1][1]
        t_n = feat_t_shapes[-1][1]
        self.simkd = SimKD_module(s_n=s_n, t_n=t_n, factor=cfg.SIMKD.FACTOR)  
        self.criterion = nn.MSELoss()
        self.cls_t =  self.teacher.get_fc_layers()
    
    def forward_train(self, image, target, **kwargs):
        # feature
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)
            
        
        # 
        trans_feat_s, trans_feat_t, pred_feat_s = self.simkd(feature_student["feats"][-1], feature_teacher["feats"][-1], self.cls_t)
        logit_s = pred_feat_s
        
        loss_feature = self.criterion(trans_feat_s, trans_feat_t)
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logit_s, target)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feature,
        }

        return logit_s, losses_dict
    
    def forward_test(self, image):
        with torch.no_grad():
            _,feat_s = self.student(image)
            _,feat_t = self.teacher(image)
            feat_t = [f.detach() for f in feat_t["feats"]]
            cls_t =  self.teacher.get_fc_layers()
            _, _, output = self.simkd(feat_s["feats"][-1], feat_t[-1], cls_t)
        return output
    def get_extra_parameters(self):
        num_p = 0
        for p in self.simkd.parameters():
            num_p += p.numel()
        return num_p
    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()] + [v for k,v in self.simkd.named_parameters()]


