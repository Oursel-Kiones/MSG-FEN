import torch
from modeling.deeplab import DeepLab  # 替换成你的模型定义
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt  # 导入 matplotlib

# 1. 定义你的模型结构 (确保与训练时一致)
model = DeepLab(num_classes=19, backbone='resnet', output_stride=16)  # 替换成你的模型参数

# 2. 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. 加载模型的状态字典
checkpoint = torch.load('/home/aistudio/data/data320849/model_best.pth.tar', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])

# 4. 设置模型为评估模式
model.eval()  # 非常重要!  这会关闭 dropout 和 batch normalization 的训练行为

# 5. 准备你的输入图像
img_path = '/home/aistudio/30000-115.9531701-2014-副本1.jpg'  # 替换成你的图像路径
img = Image.open(img_path).convert('RGB')

# 6. 图像预处理 (与训练时保持一致)
preprocess = transforms.Compose([
    transforms.Resize((513, 513)),  # 替换成你的图像尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 替换成你的归一化参数
])
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 添加 batch 维度

# 7. 推理
with torch.no_grad():  # 禁用梯度计算
    input_batch = input_batch.to(device)
    output = model(input_batch)

# 8. 处理输出
output = output.argmax(dim=1).cpu().numpy()  # 获取预测结果
# output 现在是一个 numpy 数组，包含了每个像素的类别标签

# 9. (可选) 可视化结果 并保存
plt.imshow(output[0])

# 添加保存图片的路径
save_path = '/home/aistudio/保存/prediction.png'  # 指定保存路径和文件名
plt.savefig(save_path)  # 保存图片

plt.show() # 显示图像 (可选)

print(f"Prediction done! Image saved to {save_path}")  # 修改后的输出信息