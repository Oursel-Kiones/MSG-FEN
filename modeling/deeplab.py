import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Type, List, Iterator

# 导入项目中已有的模块
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

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
    This version includes optimizations for forward pass efficiency and code maintainability.
    """

    def __init__(self, backbone: str = 'resnet', output_stride: int = 16,
                 num_classes: int = 19,
                 num_stuff_classes: int = 7,
                 num_object_classes: int = 12,
                 sync_bn: bool = True, freeze_bn: bool = False):
        super().__init__()
        
        if backbone == 'drn':
            output_stride = 8
        elif output_stride not in [8, 16]:
            raise ValueError(f"Unsupported output_stride: {output_stride} for backbone: {backbone}. Expected 8 or 16.")

        BatchNorm: Type[nn.Module] = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        
        # Note: Decoder is expected to output a 256-channel feature map.
        decoder_feature_channels = 256
        self.decoder = build_decoder(decoder_feature_channels, backbone, BatchNorm)

        self.head_semantic = nn.Conv2d(decoder_feature_channels, num_classes, kernel_size=1, stride=1)
        self.head_stuff = nn.Conv2d(decoder_feature_channels, num_stuff_classes, kernel_size=1, stride=1)
        self.head_object = nn.Conv2d(decoder_feature_channels, num_object_classes, kernel_size=1, stride=1)
        self.head_objectness = nn.Conv2d(decoder_feature_channels, 1, kernel_size=1, stride=1)
        
        self.freeze_bn_flag = freeze_bn # Renamed attribute to avoid conflict with method

    def forward(self, input: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multi-task DeepLab model. Optimized for efficiency.

        Args:
            input (torch.Tensor): Input image tensor (N, C, H, W).
            return_features (bool): If True, returns a dictionary of intermediate 
                                    features alongside the final logits. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions. The content depends on `return_features`.
        """
        input_size = input.size()[2:]

        x, low_level_feat = self.backbone(input)
        aspp_features = self.aspp(x)
        feature_map = self.decoder(aspp_features, low_level_feat)

        logits = {
            'semantic': self.head_semantic(feature_map),
            'stuff': self.head_stuff(feature_map),
            'object': self.head_object(feature_map),
            'objectness': self.head_objectness(feature_map)
        }

        if return_features:
            return {
                'logits': logits,
                'aspp_features': aspp_features,
                'low_level_feat': low_level_feat
            }
        else:
            # === REFACTOR 1: Efficient Upsampling ===
            # Concatenate all logits, perform one interpolation, then split.
            # This is more computationally efficient than multiple interpolations.
            logit_order = ['semantic', 'stuff', 'object', 'objectness']
            combined_logits = torch.cat([logits[key] for key in logit_order], dim=1)
            upsampled_combined = F.interpolate(combined_logits, size=input_size, mode='bilinear', align_corners=True)

            predictions: Dict[str, torch.Tensor] = {}
            current_channel = 0
            for key in logit_order:
                num_channels = logits[key].size(1)
                predictions[key] = upsampled_combined[:, current_channel : current_channel + num_channels, :, :]
                current_channel += num_channels
            
            return predictions

    # === REFACTOR 2: Renamed method to avoid conflict with attribute ===
    def freeze_batch_norm(self):
        """Freezes BatchNorm layers by setting them to evaluation mode."""
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

    # === REFACTOR 3: Code De-duplication for LR param getters ===
    def _get_trainable_params(self, modules: List[nn.Module]) -> Iterator[nn.Parameter]:
        """
        A helper function to extract trainable parameters from a list of modules.
        Respects the `freeze_bn_flag` attribute.
        """
        for module in modules:
            for m in module.modules(): # Use .modules() to iterate through all sub-modules
                if isinstance(m, (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                    if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)) and self.freeze_bn_flag:
                        continue # Skip BN params if freeze_bn_flag is True
                    for p in m.parameters():
                        if p.requires_grad:
                            yield p

    def get_1x_lr_params(self) -> Iterator[nn.Parameter]:
        """
        Returns parameters that should use 1x learning rate (backbone).
        """
        return self._get_trainable_params([self.backbone])

    def get_10x_lr_params(self) -> Iterator[nn.Parameter]:
        """
        Returns parameters that should use 10x learning rate (ASPP, decoder, heads).
        """
        modules_10x = [self.aspp, self.decoder, self.head_semantic, self.head_stuff,
                       self.head_object, self.head_objectness]
        return self._get_trainable_params(modules_10x)


if __name__ == "__main__":
    print("--- DeepLab Multi-Task Model Test (Optimized Version) ---")

    _NUM_CLASSES = 19
    _NUM_STUFF_CLASSES = 7
    _NUM_OBJECT_CLASSES = 12

    print(f"Initializing DeepLab with num_classes={_NUM_CLASSES}, "
          f"num_stuff_classes={_NUM_STUFF_CLASSES}, num_object_classes={_NUM_OBJECT_CLASSES}...")
    try:
        model = DeepLab(
            backbone='resnet',
            output_stride=16,
            num_classes=_NUM_CLASSES,
            num_stuff_classes=_NUM_STUFF_CLASSES,
            num_object_classes=_NUM_OBJECT_CLASSES,
            sync_bn=False # Set to False for single-GPU test
        )
        model.eval()

        print(f"Model successfully initialized.")
        
        batch_size = 2
        input_height, input_width = 513, 513
        dummy_input = torch.randn(batch_size, 3, input_height, input_width)
        print(f"\nInput tensor shape: {dummy_input.shape}")

        print("Performing forward pass (default behavior)...")
        with torch.no_grad():
            predictions = model(dummy_input)

        print("\nDefault forward pass complete. Output predictions:")
        for key, value in predictions.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # --- Verification for default forward pass ---
        assert predictions['semantic'].size(1) == _NUM_CLASSES
        assert predictions['stuff'].size(1) == _NUM_STUFF_CLASSES
        assert predictions['object'].size(1) == _NUM_OBJECT_CLASSES
        assert predictions['objectness'].size(1) == 1
        assert predictions['semantic'].shape[2:] == (input_height, input_width)
        print("\n[成功] 默认前向传播的输出通道数及分辨率均正确！")

        print("\nPerforming forward pass with return_features=True...")
        with torch.no_grad():
            features = model(dummy_input, return_features=True)

        print("\nFeature forward pass complete. Outputs:")
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
        
        # --- Verification for feature forward pass ---
        assert 'logits' in features and 'aspp_features' in features and 'low_level_feat' in features
        assert features['logits']['semantic'].size(1) == _NUM_CLASSES
        assert features['aspp_features'].size(1) == 256 # For ResNet
        print("\n[成功] 特征返回模式的输出内容和结构均正确！")


        print("\n--- DeepLab Multi-Task Model Test Completed Successfully ---")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()