import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class F1Meter(object):
    """Computes and stores the tp,fp,fn,conf_matrix and f1 value"""
    
    def __init__(self,num_classes):
        self.reset(num_classes)

    def reset(self,num_classes):
        self.tp = np.zeros(num_classes)
        self.fp = np.zeros(num_classes)
        self.fn = np.zeros(num_classes)
        self.conf_matrix = np.zeros((num_classes,num_classes))
        self.f1 = 0
    def update(self,tp,fp,fn,conf_matrix):
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.conf_matrix += conf_matrix
        self.f1 = self.calculate_f1()
    def calculate_f1(self,epsilon=1e-7):
        with torch.no_grad():     
            precision = self.tp / (self.tp + self.fp + epsilon)
            recall = self.tp / (self.tp + self.fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            f1 = f1.mean() * 100
            return f1.item()
    

def validate(val_loader, distiller,num_classes):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    f1 = F1Meter(num_classes)
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1,acc5 = accuracy(output, target, topk=(1,5))

            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0].item(), batch_size)
            top5.update(acc5[0].item(), batch_size)
            TP, FP, FN, conf_matrix = tp_fp_fn(output, target, num_classes)
            f1.update(TP, FP, FN, conf_matrix)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}| F1:{f1.f1:.3f} ".format(
                top1=top1, top5 = top5, f1=f1
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, f1.f1, losses.avg, top5.avg


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calculate_f1(output,target,num_classes,epsilon=1e-7):
    with torch.no_grad():
        output = torch.argmax(output,dim=1)
        TP, FP, FN,_ = tp_fp_fn(output, target, num_classes)
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        f1 = f1.mean() * 100
        return f1

def tp_fp_fn(preds,targets,num_classes):
    # 
    preds = torch.argmax(preds,dim=1)
    conf_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    # TP, FP, FN
    TP = torch.diag(conf_matrix)
    FP = conf_matrix.sum(0) - TP
    FN = conf_matrix.sum(1) - TP

    return TP.cpu().numpy(), FP.cpu().numpy(), FN.cpu().numpy(),conf_matrix.cpu().numpy()
def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
