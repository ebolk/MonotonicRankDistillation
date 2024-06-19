import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif "opensarship" in cfg.DATASET.TYPE:
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=True)
            if "ConvNext" in cfg.DISTILLER.STUDENT :
                model_student.head = nn.Sequential(
                    nn.BatchNorm1d(model_student.head.in_features),
                    nn.Linear(model_student.head.in_features, num_classes)
                )   
            else:
                model_student.fc = nn.Sequential(
                        nn.BatchNorm1d(model_student.fc.in_features),
                        nn.Linear(model_student.fc.in_features, num_classes)
                    )   
                # freeze parameters
                model_student.conv1.requires_grad_(False)
                model_student.bn1.requires_grad_(False)
            # for param in model_student.layer1.parameters():
            #     param.requires_grad_(False)
        elif "fusarship" in cfg.DATASET.TYPE:
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=True)
            if "ConvNext" in cfg.DISTILLER.STUDENT :
                model_student.head = nn.Sequential(
                    nn.BatchNorm1d(model_student.head.in_features),
                    nn.Linear(model_student.head.in_features, num_classes)
                ) 
            else:
                model_student.fc = nn.Sequential(
                        nn.BatchNorm1d(model_student.fc.in_features)
                        ,nn.Linear(model_student.fc.in_features, num_classes)
                    )   
            # change the first layer to apply the single channel image
            # if "Res2Net" in cfg.DISTILLER.STUDENT:
            #     model_student.conv0 = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
            # else:
            #     model_student.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif "opensarship" in cfg.DATASET.TYPE  :
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=False)
            if "ConvNext" in cfg.DISTILLER.STUDENT :
                model_teacher.head = nn.Sequential(
                    nn.BatchNorm1d(model_teacher.head.in_features),
                    nn.Linear(model_teacher.head.in_features, num_classes)
                ) 
            else:
                model_teacher.fc = nn.Sequential(
                        nn.BatchNorm1d(model_teacher.fc.in_features)
                        ,nn.Linear(model_teacher.fc.in_features, num_classes)
                    )   
            model_teacher.load_state_dict(load_checkpoint(cfg.DISTILLER.TEACHER_PATH)["model"])
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=True)
            if "ConvNext" in cfg.DISTILLER.STUDENT :
                model_student.head = nn.Sequential(
                    nn.BatchNorm1d(model_student.head.in_features),
                    nn.Linear(model_student.head.in_features, num_classes)
                ) 
            else:
                model_student.fc = nn.Sequential(
                        nn.BatchNorm1d(model_student.fc.in_features)
                        ,nn.Linear(model_student.fc.in_features, num_classes)
                    )   
                model_student.conv1.requires_grad_(False)
                model_student.bn1.requires_grad_(False)
        elif cfg.DATASET.TYPE == "fusarship":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=False)
            model_teacher.fc = nn.Sequential(
                    nn.BatchNorm1d(model_teacher.fc.in_features)
                    ,nn.Linear(model_teacher.fc.in_features, num_classes)
                )
            # model_teacher.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)   # ，
            model_teacher.load_state_dict(load_checkpoint(cfg.DISTILLER.TEACHER_PATH)["model"])
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=True)
            model_student.fc = nn.Sequential(
                    nn.BatchNorm1d(model_student.fc.in_features)
                    ,nn.Linear(model_student.fc.in_features, num_classes)
                )   
            # model_student.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg,num_classes
    )
    trainer.train(resume=resume)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    import random
    random.seed(seed)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    # fix the random seed
    seed = 42
    # 7
    setup_seed(seed)
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # only use single gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    # close ssl
    import ssl
    ssl.__create_default_https_context = ssl._create_unverified_context
    
    device_num = torch.cuda.device_count()
    # batch size
    if device_num > 1:
        cfg.SOLVER.BATCH_SIZE *= device_num
        cfg.DATASET.TEST.BATCH_SIZE *= device_num
        cfg.SOLVER.LR *= device_num
    cfg.freeze()
    # torch.autograd.set_detect_anomaly(True)
    main(cfg, args.resume, args.opts)
