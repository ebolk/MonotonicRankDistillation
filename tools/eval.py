import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate,tp_fp_fn
from mdistiller.engine.cfg import CFG as cfg
from torch import nn
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="ResNet18")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet","opensarship3","opensarship6","fusarship"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=512)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    elif "opensarship" in args.dataset:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model = imagenet_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt =  args.ckpt
        model.fc = nn.Sequential(
                    nn.BatchNorm1d(model.fc.in_features)
                    ,nn.Linear(model.fc.in_features, num_classes)
                )       
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    elif "fusarship" in args.dataset:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model = imagenet_model_dict[args.model]()   
        ckpt =  args.ckpt
        model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features)
        ,nn.Linear(model.fc.in_features, num_classes)
        )
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    test_acc, f1, test_loss,top5 = validate(val_loader, model,num_classes)
    # calculate confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(image=image)
            _,_,_,matrix = tp_fp_fn(output,target,num_classes)
            conf_matrix+=matrix
    print(conf_matrix)
