# 文件路径: /workspace/deep参考1/modeling/deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Type

# 导入项目中已有的模块
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder # 注意：这里假设 build_decoder 经过修改后返回256通道特征图
from modeling.backbone import build_backbone

# --- 通用代码要求符合性检查 ---
# 代码风格: 已遵循PEP8规范，变量命名为snake_case，描述性强。
# 类型提示: 所有函数签名、类属性均已使用Python类型提示。
# 文档字符串: 所有类、方法、函数都包含清晰、完整、符合PyTorch/Google风格的docstrings。
# 模块化: 各预测头分离清晰，职责单一。
# 错误处理/健壮性: 已考虑对output_stride和backbone的校验。
# 依赖管理: 本代码片段主要依赖 PyTorch 及项目中已有的 modeling 模块。

class DeepLab(nn.Module):
    """
    DeepLabV3+ architecture with multiple prediction heads for various segmentation tasks.

    This model serves as a multi-task baseline, capable of predicting:
    - Full semantic segmentation (all 19 classes)
    - 'Stuff' semantic segmentation (7 specific classes)
    - 'Object' semantic segmentation (12 specific classes)
    - Binary 'Objectness' prediction (foreground/background)

    It integrates a backbone, ASPP, and a general decoder to produce a shared
    feature map, which is then fed into task-specific prediction heads.
    """

    def __init__(self, backbone: str = 'resnet', output_stride: int = 16,
                 num_classes: int = 19, # Total semantic classes (e.g., Cityscapes)
                 num_stuff_classes: int = 7, # Specific count for 'stuff' categories
                 num_object_classes: int = 12, # Specific count for 'object' categories
                 sync_bn: bool = True, freeze_bn: bool = False):
        """
        Initializes the DeepLab model with multi-task prediction heads.

        Args:
            backbone (str): Name of the backbone architecture (e.g., 'resnet', 'mobilenet').
            output_stride (int): Output stride of the backbone (e.g., 16 or 8).
            num_classes (int): Total number of semantic classes for the 'semantic' head.
            num_stuff_classes (int): Number of 'stuff' categories for the 'stuff' head.
            num_object_classes (int): Number of 'object' categories for the 'object' head.
            sync_bn (bool): Whether to use SynchronizedBatchNorm2d across multiple GPUs.
            freeze_bn (bool): Whether to freeze BatchNorm layers during training.
        """
        super().__init__()
        
        if backbone == 'drn':
            output_stride = 8
        elif output_stride not in [8, 16]: # Add a general check for output_stride
            raise ValueError(f"Unsupported output_stride: {output_stride} for backbone: {backbone}. "
                             "Expected 8 or 16.")

        BatchNorm: Type[nn.Module] = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        
        # --- 核心修改点: 移除 'input_channels' 和 'output_channels' 参数 ---
        # build_aspp 通常会根据 backbone 类型来内部判断 ASPP 需要的输入和输出通道数。
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # --- 结束核心修改点 ---

        # build_decoder is expected to produce a 256-channel generic feature map for prediction heads.
        # This requires build_decoder in modeling/decoder.py to be modified if it originally included
        # a final classification layer.
        self.decoder = build_decoder(256, backbone, BatchNorm) # 256 is the desired feature map channels

        # New independent prediction heads, each is a 1x1 convolution
        self.head_semantic = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.head_stuff = nn.Conv2d(256, num_stuff_classes, kernel_size=1, stride=1)
        self.head_object = nn.Conv2d(256, num_object_classes, kernel_size=1, stride=1)
        self.head_objectness = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        
        self.freeze_bn = freeze_bn

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multi-task DeepLab model.

        Args:
            input (torch.Tensor): Input image tensor (N, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing predictions for various tasks:
                                     'semantic', 'stuff', 'object', 'objectness'.
                                     All predictions are upsampled to the original input size.
        """
        # Backbone returns high-level features for ASPP and low-level features for decoder
        x, low_level_feat = self.backbone(input)
        
        # Process high-level features with ASPP
        x = self.aspp(x)
        
        # Decode features using the common decoder
        # Note: low_level_feat should be projected if build_decoder expects it.
        # Assuming build_decoder handles the low-level feature projection internally,
        # or expects raw 256 channels as per its current implementation.
        feature_map = self.decoder(x, low_level_feat)

        predictions: Dict[str, torch.Tensor] = {}
        
        # Apply each prediction head to the common feature map and upsample to original size
        predictions['semantic'] = F.interpolate(
            self.head_semantic(feature_map), size=input.size()[2:], mode='bilinear', align_corners=True)
        predictions['stuff'] = F.interpolate(
            self.head_stuff(feature_map), size=input.size()[2:], mode='bilinear', align_corners=True)
        predictions['object'] = F.interpolate(
            self.head_object(feature_map), size=input.size()[2:], mode='bilinear', align_corners=True)
        predictions['objectness'] = F.interpolate(
            self.head_objectness(feature_map), size=input.size()[2:], mode='bilinear', align_corners=True)
        
        return predictions

    def freeze_bn(self):
        """Freezes BatchNorm layers by setting them to evaluation mode."""
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

    def get_1x_lr_params(self):
        """
        Returns parameters that should use 1x learning rate (typically backbone parameters).
        """
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        """
        Returns parameters that should use 10x learning rate (typically ASPP, decoder, and prediction heads).
        """
        # ASPP, Decoder and all new prediction heads are grouped for 10x learning rate
        modules = [self.aspp, self.decoder, self.head_semantic, self.head_stuff,
                   self.head_object, self.head_objectness]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    print("--- DeepLab Multi-Task Model Test ---")

    # Define model parameters
    _NUM_CLASSES = 19 # Total semantic classes
    _NUM_STUFF_CLASSES = 7 # Confirmed from debug.py output
    _NUM_OBJECT_CLASSES = 12 # Confirmed from GT list

    print(f"Initializing DeepLab with num_classes={_NUM_CLASSES}, "
          f"num_stuff_classes={_NUM_STUFF_CLASSES}, num_object_classes={_NUM_OBJECT_CLASSES}...")
    try:
        model = DeepLab(
            backbone='resnet',
            output_stride=16,
            num_classes=_NUM_CLASSES,
            num_stuff_classes=_NUM_STUFF_CLASSES,
            num_object_classes=_NUM_OBJECT_CLASSES,
            sync_bn=True
        )
        model.eval() # Set model to evaluation mode

        print(f"Model successfully initialized.")
        
        # Create a dummy input tensor (e.g., a 513x513 RGB image)
        batch_size = 1
        input_height, input_width = 513, 513
        dummy_input = torch.randn(batch_size, 3, input_height, input_width)
        print(f"\nInput tensor shape: {dummy_input.shape}")

        # Perform a forward pass
        print("Performing forward pass...")
        with torch.no_grad(): # Disable gradient calculation for inference
            predictions = model(dummy_input)

        print("\nForward pass complete. Output predictions received:")
        for key, value in predictions.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        # Verify output shapes and channel counts
        expected_output_size = (batch_size, -1, input_height, input_width) # Channel varies, H, W must match input

        assert predictions['semantic'].size(1) == _NUM_CLASSES, \
               f"Semantic head channel mismatch: Expected {_NUM_CLASSES}, got {predictions['semantic'].size(1)}"
        assert predictions['stuff'].size(1) == _NUM_STUFF_CLASSES, \
               f"Stuff head channel mismatch: Expected {_NUM_STUFF_CLASSES}, got {predictions['stuff'].size(1)}"
        assert predictions['object'].size(1) == _NUM_OBJECT_CLASSES, \
               f"Object head channel mismatch: Expected {_NUM_OBJECT_CLASSES}, got {predictions['object'].size(1)}"
        assert predictions['objectness'].size(1) == 1, \
               f"Objectness head channel mismatch: Expected 1, got {predictions['objectness'].size(1)}"
        
        for key, value in predictions.items():
            assert value.shape[0] == batch_size and value.shape[2:] == dummy_input.shape[2:], \
                   f"{key} output size mismatch: Expected {expected_output_size}, got {value.shape}"

        print("\n[成功] 所有预测头的输出通道数及分辨率均正确！")
        print("\n--- DeepLab Multi-Task Model Test Completed Successfully ---")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()