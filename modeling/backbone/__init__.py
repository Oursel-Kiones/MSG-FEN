# /workspace/deep参考1/modeling/backbone/__init__.py

import torch.nn as nn
from typing import Type

# 确保这些模块和类是真实存在的
from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone_name: str, output_stride: int, BatchNorm: Type[nn.Module]) -> nn.Module:
    """构建一个指定名称的骨干网络 (backbone)。

    这是一个工厂函数，根据输入的名称选择并实例化相应的骨干网络模型。

    Args:
        backbone_name (str): 骨干网络的名称。
            支持的值: 'resnet', 'xception', 'drn', 'mobilenet' (不区分大小写)。
        output_stride (int): 网络的期望输出步长。通常为 8 或 16。
        BatchNorm (Type[nn.Module]): 要使用的批归一化层。
            例如 `nn.BatchNorm2d` 或 `modeling.sync_batchnorm.SynchronizedBatchNorm2d`。

    Returns:
        nn.Module: 一个实例化的 PyTorch 模型，作为骨干网络。
        该模型的前向传播 (`forward`) 方法被设计为返回两个张量：
        - high_level_features (高阶特征, for ASPP)
        - low_level_features (低阶特征, for Decoder)

    Raises:
        NotImplementedError: 如果 `backbone_name` 不是支持的类型。
    """
    # 将输入名称转为小写，以增加健壮性
    backbone_name_lower = backbone_name.lower()

    # 根据名称选择并返回相应的模型实例
    # 注意：我们直接将参数传递给模型类的构造函数
    if backbone_name_lower == 'resnet' or backbone_name_lower == 'resnet101':
        return resnet.ResNet101(output_stride=output_stride, BatchNorm=BatchNorm)
    elif backbone_name_lower == 'xception':
        return xception.AlignedXception(output_stride=output_stride, BatchNorm=BatchNorm)
    elif backbone_name_lower == 'drn':
        return drn.drn_d_54(BatchNorm=BatchNorm)
    elif backbone_name_lower == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride=output_stride, BatchNorm=BatchNorm)
    else:
        # 抛出带有详细信息的错误，方便调试
        raise NotImplementedError(
            f"Backbone '{backbone_name}' is not supported. "
            f"Supported backbones are: 'resnet', 'xception', 'drn', 'mobilenet'."
        )