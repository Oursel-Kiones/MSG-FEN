import unittest
import torch
import sys
import os

# 将项目根目录添加到 Python 路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling.deeplab import DeepLab
from modeling.msg_fenet import MSG_FENet_Stage2

class TestStage2Integration(unittest.TestCase):
    """
    端到端集成测试：验证 MSG_FENet_Stage2 与 DeepLab 的协同工作。
    """
    
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前运行一次，用于准备耗时的资源。"""
        print("\n" + "="*50)
        print("= 正在设置 Stage 2 集成测试套件 =")
        print("="*50)

        cls.NUM_STUFF_CLASSES = 7
        cls.NUM_OBJECT_CLASSES = 12
        cls.NUM_TOTAL_CLASSES = 19

        print("\n[步骤 1] 正在初始化 Stage 1 DeepLab 模型...")
        stage1_model = DeepLab(
            backbone='resnet', 
            output_stride=16, 
            num_classes=cls.NUM_TOTAL_CLASSES,
            num_stuff_classes=cls.NUM_STUFF_CLASSES,
            num_object_classes=cls.NUM_OBJECT_CLASSES,
            sync_bn=False
        )
        print(" -> Stage 1 模型初始化成功。")

        print("\n[步骤 2] 正在初始化 Stage 2 MSG-FENet 模型...")
        cls.model = MSG_FENet_Stage2(
            stage1_model=stage1_model, 
            num_thing_classes=cls.NUM_OBJECT_CLASSES
        )
        print(" -> Stage 2 模型初始化成功。")
        
        if torch.cuda.is_available():
            cls.device = 'cuda'
            print("\n[信息] CUDA 可用。正在将模型移至 GPU。")
        else:
            cls.device = 'cpu'
            print("\n[信息] CUDA 不可用。正在 CPU 上运行测试。")
        cls.model.to(cls.device)
        
    def test_forward_pass_and_output_shapes(self):
        """测试 1: 验证前向传播是否能成功执行，以及输出的形状是否正确。"""
        print("\n--- 正在运行测试 1: 前向传播与输出形状 ---")
        dummy_input = torch.randn(2, 3, 512, 1024).to(self.device)
        print(f"输入张量形状: {dummy_input.shape}")
        
        with torch.no_grad():
            self.model.eval()
            output = self.model(dummy_input)
        
        print(f"输出字典的键: {list(output.keys())}")
        
        self.assertIsInstance(output, dict)
        self.assertIn('things', output)
        self.assertIn('stuff', output)
        
        expected_things_shape = (2, self.NUM_OBJECT_CLASSES, 512, 1024)
        expected_stuff_shape = (2, self.NUM_STUFF_CLASSES, 512, 1024)
        self.assertEqual(output['things'].shape, expected_things_shape)
        self.assertEqual(output['stuff'].shape, expected_stuff_shape)
        
        print("✅ 成功: 前向传播已执行，输出形状正确。")
        
    def test_backward_pass_and_gradient_flow(self):
        """测试 2 (关键): 验证反向传播和梯度流。"""
        print("\n--- 正在运行测试 2: 反向传播与梯度流 ---")
        
        dummy_input = torch.randn(2, 3, 512, 1024).to(self.device)
        self.model.train()
        
        output = self.model(dummy_input)
        
        loss = output['things'].mean()
        loss.backward()
        print(f"已计算伪损失并执行反向传播。")

        # 验证可训练部分的梯度
        fusion_grad = self.model.feature_fusion.conv_block[0].weight.grad
        decoder_grad = self.model.decoder_object.last_conv[0].weight.grad
        self.assertIsNotNone(fusion_grad, "FeatureFusionModule 的梯度应该存在！")
        self.assertIsNotNone(decoder_grad, "Decoder_Object 的梯度应该存在！")
        self.assertGreater(torch.abs(fusion_grad).sum(), 0)
        self.assertGreater(torch.abs(decoder_grad).sum(), 0)
        print("✅ 成功: Stage 2 的可训练参数 (Fusion, Decoder) 均有梯度。")

        # === 关键修复点：使用更健壮的方式检查冻结参数的梯度 ===
        # 我们不关心参数的具体名字，只关心模块的第一个参数是否有梯度。
        # 旧的脆弱方法: self.model.stage1_model.backbone.conv1.weight.grad
        # 新的健壮方法: next(self.model.stage1_model.backbone.parameters()).grad
        stage1_backbone_params = list(self.model.stage1_model.backbone.parameters())
        stage1_aspp_params = list(self.model.stage1_model.aspp.parameters())

        self.assertTrue(len(stage1_backbone_params) > 0, "Stage 1 backbone should have parameters.")
        self.assertTrue(len(stage1_aspp_params) > 0, "Stage 1 ASPP should have parameters.")
        
        stage1_backbone_grad = stage1_backbone_params[0].grad
        stage1_aspp_grad = stage1_aspp_params[0].grad
        
        self.assertIsNone(stage1_backbone_grad, "Stage 1 backbone 的梯度不应该存在！")
        self.assertIsNone(stage1_aspp_grad, "Stage 1 ASPP 的梯度不应该存在！")
        print("✅ 成功: Stage 1 的冻结参数 (Backbone, ASPP) 均无梯度。")

if __name__ == '__main__':
    print("正在开始 MSG-FENet Stage 2 的集成测试...")
    unittest.main()