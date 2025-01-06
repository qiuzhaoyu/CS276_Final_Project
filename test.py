import os
import torch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置 TORCH_HOME 环境变量，将缓存目录指定为当前目录的 .cache 文件夹
os.environ['TORCH_HOME'] = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache'

# 如果 GPU 不可用，选择禁用 xFormers 优化
if device == torch.device("cpu"):
    os.environ["XFORMERS_FORCE_DISABLE"] = "1"
    print("Running on CPU, disabling xFormers optimizations.")

# 加载 DINOv2 模型
print("Loading DINOv2 model...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# 将模型参数转换为 float16 并移动到 GPU 或 CPU
if device.type == "cuda":
    model = model.half()  # 转换模型参数为 float16
model = model.to(device)
model.eval()

# 示例输入张量 (假设输入为 3x224x224 的 RGB 图像)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 如果运行在 GPU 上，转换输入张量为 float16
if device.type == "cuda":
    dummy_input = dummy_input.half()  # 转换为 float16
    print("Input tensor converted to float16 for GPU acceleration.")

# 运行模型并获取输出
try:
    with torch.no_grad():
        output = model(dummy_input)
    print("Model output shape:", output.shape)
except RuntimeError as e:
    print("Error during model execution:", e)
