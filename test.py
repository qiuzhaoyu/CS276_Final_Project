import torch
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# 加载 DINOv2 预训练模型
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# 切换到评估模式
model.eval()

# 示例输入张量 (假设输入为 3x224x224 图像)
dummy_input = torch.randn(1, 3, 224, 224)

# 获取模型输出
with torch.no_grad():
    output = model(dummy_input)

print(output.shape)  # 输出特征嵌入
