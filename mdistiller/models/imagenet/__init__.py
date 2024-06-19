from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv1 import MobileNetV1
from .res2net  import res2net101_v1b,res2net50_v1b
from .convnext import convnext_base,convnext_tiny
from .resnetv2 import resnetv18,resnetv34

imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "MobileNetV1": MobileNetV1,
    "Res2Net101": res2net101_v1b,
    "Res2Net50": res2net50_v1b,
    "ConvNextTiny": convnext_tiny,
    "ConvNextBase": convnext_base,
    "ResNetv18": resnetv18,
    "ResNetv34": resnetv34
    
}
