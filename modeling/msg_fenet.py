# 文件路径: /workspace/deep参考1/modeling/msg_fenet.py
print("--- MODEL VERSION CHECK: MSG-FENet Stage 2 (End-to-End Unfrozen Version) ---")

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    """
    专属的特征融合模块。
    将来自 ASPP 的高层特征与 Stage 1 的 objectness 预测图进行融合。
    """
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(FeatureFusionModule, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, high_level_features, objectness_logit):
        # 确保 objectness_logit 的尺寸与 high_level_features 一致
        if objectness_logit.shape[2:] != high_level_features.shape[2:]:
            objectness_logit = F.interpolate(
                objectness_logit, size=high_level_features.shape[2:], 
                mode='bilinear', align_corners=True
            )
        # 拼接：高层语义 (256) + 物体性先验 (1) = 257 通道
        fused_input = torch.cat([high_level_features, objectness_logit], dim=1)
        fused_output = self.conv_block(fused_input)
        return fused_output

class Decoder_Object(nn.Module):
    """
    专属的物体解码器 (Stage 2)。
    接收融合特征与底层特征的拼接结果，解码为最终的 12 类 Object 分割图。
    """
    def __init__(self, in_channels, num_classes, BatchNorm=nn.BatchNorm2d):
        super(Decoder_Object, self).__init__()
        # 模仿 DeepLabV3+ 的 decoder 结构
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            # 最终输出 num_classes (12类)
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, input_size):
        x = self.last_conv(x)
        # 上采样回原始图像分辨率 (如 513x513)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

class MSG_FENet_Stage2(nn.Module):
    """
    【核心类】MSG-FENet Stage 2 总装配平台 (端到端解封版)
    采用组合 (Composition) 和 Hook 技术。解封 Backbone 以突破 mIoU 瓶颈。
    """
    def __init__(self, stage1_model, num_object_classes=12, BatchNorm=nn.BatchNorm2d):
        super(MSG_FENet_Stage2, self).__init__()
        
        # 1. 装载 Stage 1 引擎 (解封：不再冻结 requires_grad)
        self.stage1_model = stage1_model
        print("[MSG-FENet] Unfreezing Stage 1 model for End-to-End fine-tuning...")
        # 注意：这里我们删除了 param.requires_grad = False 和 eval()，允许网络更新
        
        # 2. 准备“传感器”数据容器
        self.intermediate_features = {}
        
        # 3. 定义并挂载 Hooks (钩子)
        def get_backbone_hook(name):
            def hook(model, input, output):
                self.intermediate_features[name] = output[1]
            return hook
            
        def get_aspp_hook(name):
            def hook(model, input, output):
                self.intermediate_features[name] = output
            return hook
            
        # 挂载传感器！
        self.stage1_model.backbone.register_forward_hook(get_backbone_hook('low_level'))
        self.stage1_model.aspp.register_forward_hook(get_aspp_hook('aspp_feat'))

        # ==========================================
        # 4. 构建 Stage 2 专属的新模块
        # ==========================================
        aspp_channels = 256
        low_level_channels = 256 # ResNet 的 low_level_feat 通常是 256
        compress_channels = 48   # DeepLab 降维标准
        
        # 1x1 卷积：压缩低层特征
        self.low_level_compressor = nn.Sequential(
            nn.Conv2d(low_level_channels, compress_channels, kernel_size=1, bias=False),
            BatchNorm(compress_channels),
            nn.ReLU()
        )
        
        # 融合模块: ASPP (256) + Objectness (1)
        self.feature_fusion = FeatureFusionModule(in_channels=aspp_channels + 1, 
                                                  out_channels=aspp_channels, 
                                                  BatchNorm=BatchNorm)
        
        # 物体解码器: 融合高层特征 (256) + 压缩低层特征 (48)
        self.decoder_object = Decoder_Object(in_channels=aspp_channels + compress_channels,
                                             num_classes=num_object_classes,
                                             BatchNorm=BatchNorm)

    def forward(self, input):
        input_size = input.size()[2:] # H, W (例如 513x513)
        
        # ==========================================
        # 步骤 A: 运行引擎 (连通计算图，允许梯度回传)
        # ==========================================
        # 去掉了 with torch.no_grad()，此时梯度可以流回 Backbone
        stage1_predictions = self.stage1_model(input)
            
        # ==========================================
        # 步骤 B: 提取截获的数据
        # ==========================================
        aspp_feat = self.intermediate_features['aspp_feat']
        low_level_feat = self.intermediate_features['low_level']
        logit_objectness = stage1_predictions['objectness']
        logit_stuff = stage1_predictions['stuff']

        # ==========================================
        # 步骤 C: Stage 2 专属数据流
        # ==========================================
        # 1. 融合 ASPP 与 Objectness 先验
        # 【关键保护】：给 logit_objectness 加 .detach()！
        # 因为我们不想让 Stage 2 改变物体类的 Loss 倒流回去破坏 Stage 1 的二分类头。
        fused_high_level = self.feature_fusion(aspp_feat, logit_objectness.detach())
        
        # 2. 压缩低层特征 (256 -> 48)
        compressed_low_level = self.low_level_compressor(low_level_feat)
        
        # 3. 空间对齐：将高层特征上采样到低层特征的尺寸
        fused_high_level = F.interpolate(
            fused_high_level, size=compressed_low_level.size()[2:], 
            mode='bilinear', align_corners=True
        )
        
        # 4. 拼接高低层特征
        concat_feat = torch.cat((fused_high_level, compressed_low_level), dim=1)
        
        # 5. 最终解码预测 (12 类 objects)
        logit_things = self.decoder_object(concat_feat, input_size)
        
        # 返回 Stage 2 的 objects 和 Stage 1 的 stuff
        return {'object': logit_things, 'stuff': logit_stuff}