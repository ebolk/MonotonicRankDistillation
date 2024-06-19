import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from ._common import ConvReg, get_feat_shapes

def fdl_loss(logits_student, logits_teacher,phase_weight,rand):
    fft_x = torch.fft.fftn(logits_student,dim=-1)
    fft_y = torch.fft.fftn(logits_teacher,dim=-1)
    x_mag = torch.abs(fft_x)
    x_phase = torch.angle(fft_x)
    y_mag = torch.abs(fft_y)
    y_phase = torch.angle(fft_y)
    
    x_mag = torch.mm(x_mag.view(x_mag.size(0),-1),rand)
    y_mag = torch.mm(y_mag.view(y_mag.size(0),-1),rand)
    
    x_phase = torch.mm(x_phase.view(x_phase.size(0),-1),rand)
    y_phase = torch.mm(y_phase.view(y_phase.size(0),-1),rand)
    
    x_mag,_ = torch.sort(x_mag,dim=-1)
    y_mag,_ = torch.sort(y_mag,dim=-1)
    x_phase,_ = torch.sort(x_phase,dim=-1)
    y_phase,_ = torch.sort(y_phase,dim=-1)
    s_amplitude = torch.abs(x_mag - y_mag)
    s_phase = torch.abs(x_phase - y_phase)
    s = s_amplitude + s_phase * phase_weight
    return s.mean()
    

class FDL(Distiller):
    def __init__(
        self,student,teacher,cfg):
        """
        patch_size, stride, num_proj: SWD slice parameters
        model: feature extractor, support VGG, ResNet, Inception, EffNet
        phase_weight: weight for phase branch
        """
        super(FDL, self).__init__(student, teacher)

        self.phase_weight = cfg.FDL.PHASE_WEIGHT
        self.stride = cfg.FDL.STRIDE
        patch_size = cfg.FDL.PATCH_SIZE
        num_proj = cfg.FDL.NUM_PROJ
        self.ce_loss_weight = cfg.FDL.LOSS.CE_WEIGHT
        self.fdl_loss_weight = cfg.FDL.LOSS.FDL_WEIGHT
        self.num_filters = self.teacher.get_stage_channels()
        self.warmup = cfg.FDL.WARMUP

        # print all the parameters
        rand = torch.randn(100,num_proj)
        rand = rand / rand.norm(dim=-1,keepdim=True)
        self.register_buffer('rand',rand)
        
            
    def forward_train(self,image,target,**kwargs):
        logits_student,_ = self.student(image)
        with torch.no_grad():
            logits_teacher,_  = self.teacher(image)
            
        # losses
        loss_fdl =  self.fdl_loss_weight * fdl_loss(logits_student,logits_teacher,self.phase_weight,self.__getattr__('rand'))
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_fdl,
        }
        
        return logits_student,losses_dict  # the bigger the score, the bigger the difference between the two images


# if __name__ == '__main__':
#     print("FDL_loss")
#     X = torch.randn((1, 3,128,128)).cuda()
#     Y = torch.randn((1, 3,128,128)).cuda() * 2
#     loss = FDL_loss().cuda()
#     c = loss(X,Y)
#     print('loss:', c)
