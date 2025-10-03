import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Type

# 导入我们修改好并提供稳定接口的 DeepLab 类
from modeling.deeplab import DeepLab

class FeatureFusionModule(nn.Module):
    """
    专属的特征融合模块 (Feature Fusion Module)。

    职责: 实现 Stage 1 到 Stage 2 的关键信息传递。它接收两个输入：
    1. 来自 DeepLab 骨干网络的高层语义特征 (ASPP 输出)。
    2. 来自 Stage 1 输出的 objectness 概率图 (logits)。

    通过维度对齐、通道拼接和特征提炼，为 Stage 2 解码器提供信息更丰富的输入。
    """
    def __init__(self, in_channels: int, out_channels: int, BatchNorm: Type[nn.Module] = nn.BatchNorm2d):
        super(FeatureFusionModule, self).__init__()
        # 3x3 卷积块用于学习高层特征和 objectness 空间先验的交互关系
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, high_level_features: torch.Tensor, objectness_logit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            high_level_features (torch.Tensor): ASPP 输出的特征图。
            objectness_logit (torch.Tensor): Stage 1 预测的 objectness logit。

        Returns:
            torch.Tensor: 融合并提炼后的特征图。
        """
        if objectness_logit.shape[2:] != high_level_features.shape[2:]:
            objectness_logit = F.interpolate(
                objectness_logit, size=high_level_features.shape[2:], 
                mode='bilinear', align_corners=True
            )
        fused_input = torch.cat([high_level_features, objectness_logit], dim=1)
        fused_output = self.conv_block(fused_input)
        return fused_output

class Decoder_Object(nn.Module):
    """
    专属的物体解码器 (Object Decoder)。

    职责: 将经过 FeatureFusionModule 融合后的特征图，解码为最终的物体类别分割预测。
    它接收融合了低阶特征的输入，并输出 N 个物体类别的 logits。
    """
    def __init__(self, in_channels: int, num_classes: int, BatchNorm: Type[nn.Module] = nn.BatchNorm2d):
        super(Decoder_Object, self).__init__()
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        x = self.last_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

class MSG_FENet_Stage2(nn.Module):
    """
    MSG-FENet Stage 2 的主网络架构 (总装配平台)。

    通过“松耦合组合”方式，在 Stage 1 模型基础上进行扩展，专注于物体精细化分割。
    此版本经过架构优化，效率更高，更符合 DeepLabV3+ 设计。
    """
    def __init__(self, stage1_model: DeepLab, num_thing_classes: int, BatchNorm: Type[nn.Module] = nn.BatchNorm2d):
        super(MSG_FENet_Stage2, self).__init__()
        
        self.stage1_model = stage1_model
        print("[MSG-FENet] INFO: Freezing all parameters of the provided Stage 1 model...")
        for param in self.stage1_model.parameters():
            param.requires_grad = False
            
        # 定义通道数 (基于 ResNet + DeepLabV3+ 的典型配置)
        aspp_out_channels = 256
        low_level_channels_in = 256 # 来自 ResNet layer1 (C2)
        low_level_channels_out = 48 # DeepLabV3+ 论文中推荐的压缩后通道数
        
        # 融合模块输入 = ASPP特征(256) + objectness(1)
        self.feature_fusion = FeatureFusionModule(in_channels=aspp_out_channels + 1, 
                                                  out_channels=aspp_out_channels, 
                                                  BatchNorm=BatchNorm)
        
        # === 架构优化点：新增低阶特征压缩模块 ===
        # 模仿 DeepLabV3+ 的标准解码器，使用 1x1 卷积来压缩低阶特征的通道数。
        # 这可以减少解码器部分的参数量和计算量，提高效率。
        self.low_level_compressor = nn.Sequential(
            nn.Conv2d(low_level_channels_in, low_level_channels_out, 1, bias=False),
            BatchNorm(low_level_channels_out),
            nn.ReLU()
        )
        
        # 解码器输入 = 融合后上采样特征(256) + 压缩后的 low_level 特征(48)
        decoder_in_channels = aspp_out_channels + low_level_channels_out
        self.decoder_object = Decoder_Object(in_channels=decoder_in_channels,
                                             num_classes=num_thing_classes,
                                             BatchNorm=BatchNorm)

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = input.size()[2:] # H, W
        
        with torch.no_grad():
            self.stage1_model.eval()
            stage1_outputs = self.stage1_model(input, return_features=True)
        
        stage1_aspp_features = stage1_outputs['aspp_features']
        low_level_feat = stage1_outputs['low_level_feat']
        stage1_logits = stage1_outputs['logits']
        logit_objectness = stage1_logits['objectness']
        
        # === Stage 2 专属数据流 ===
        # 1. 融合 ASPP 特征和 objectness 先验
        fused_features = self.feature_fusion(stage1_aspp_features, logit_objectness)
        
        # 2. 压缩低阶特征
        low_level_feat_compressed = self.low_level_compressor(low_level_feat)

        # 3. 模仿 DeepLabV3+ 解码器，与 *压缩后* 的低阶特征融合
        x = F.interpolate(fused_features, size=low_level_feat_compressed.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat_compressed), dim=1)
        
        # 4. 通过物体解码器得到最终的 things 预测
        logit_things = self.decoder_object(x, input_size)
        
        # 5. 从 Stage 1 直接获取 stuff 预测，并上采样
        logit_stuff = F.interpolate(stage1_logits['stuff'], size=input_size, mode='bilinear', align_corners=True)
        
        return {'things': logit_things, 'stuff': logit_stuff}