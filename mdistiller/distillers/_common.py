import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=False):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
    
class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        
        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.global_T = nn.Parameter(torch.log(torch.tensor(torch.e+1)),requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)

class Mlp_T(nn.Module):
    def __init__(self, dim_in=200, dim_out=1) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, dim_in // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(dim_in // 2),
            nn.Linear(dim_in // 2, dim_out),
            nn.ReLU(),
        )

    def forward(self, x1,x2,lambda_):
        x = torch.cat([x1,x2],dim=1)
        x = x.view(x.shape[0], -1)
        out = self.layer(x)
        return out

from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        # print(grads)
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


if __name__ == '__main__':

    model = Global_T()
    input = torch.rand(24,24,24)
    input2 = torch.rand(24,24,24)

    out = model(input, input2,-1)
    out = out * 5
    out.backward()
    
    print(model.global_T.grad)