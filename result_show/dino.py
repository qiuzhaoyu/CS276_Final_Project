import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import os, random
import matplotlib.pyplot as plt
from PIL import Image

os.environ['TORCH_HOME'] = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache'

# 加载数据函数
def load_data():
    prefix = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache/covid-segmentation'
    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)
    test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)
    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg

# 定义 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.shape[-2:]
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.final_conv(x)

# 定义 DinoSegmentationModel
class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes=4, patch_size=14, feat_dim=384):
        super(DinoSegmentationModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ASPP(256, 128),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def preprocess(self, x):
        transform = T.Compose([
            T.Resize((self.patch_size * 38, self.patch_size * 38)),
            T.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        x = torch.cat([x] * 3, dim=1)  # 将单通道扩展为三通道
        return transform(x)

    def forward(self, x):
        x = self.preprocess(x)
        features_dict = self.backbone.forward_features(x)
        features = features_dict['x_norm_patchtokens']
        batch_size, num_patches, _ = features.shape
        patch_h = patch_w = int(num_patches ** 0.5)
        features = features.transpose(1, 2).reshape(batch_size, self.feat_dim, patch_h, patch_w)
        output = self.decoder(features)
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output

# 保存对比图片的函数
def save_prediction_image(image, mask, prediction, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image.squeeze(), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 真实掩膜
    axes[1].imshow(mask, cmap="jet", alpha=0.7)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    # 预测结果
    axes[2].imshow(prediction, cmap="jet", alpha=0.7)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Prediction saved at: {save_path}")

# 设置随机种子
seed = 54
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 加载数据
images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()

# 加载模型和预训练权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')  # 加载预训练的 DINOv2 模型
model = DinoSegmentationModel(backbone, num_classes=4, patch_size=14, feat_dim=384)

checkpoint_path = "/public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/best_model_DINOv2.pth"  # 替换为实际路径
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 验证集中的一张图片和真实掩膜
val_image = images_radiopedia[0]  # 替换为实际验证集图片
val_mask = masks_radiopedia[0]    # 替换为实际验证集掩膜

# 转换为张量并进行推理
test_input_tensor = torch.tensor(val_image).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 1, 512, 512]
with torch.no_grad():
    prediction_output = model(test_input_tensor)
prediction_classes = torch.argmax(prediction_output, dim=1).cpu().numpy()[0]  # 转换为 NumPy 数组 (512, 512)

# 保存对比图像
save_path = "prediction_result.png"  # 替换为实际保存路径
save_prediction_image(val_image.squeeze(), val_mask, prediction_classes, save_path)
