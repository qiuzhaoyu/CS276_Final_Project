import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import pickle
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
# # 设置 TORCH_HOME
os.environ['TORCH_HOME'] = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache'

# --------------------- 1. 指标函数 --------------------- #
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = (output == mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    with torch.no_grad():
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(n_classes):
            pred_inds = (pred_mask == clas)
            target_inds = (mask == clas)
            if target_inds.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(pred_inds, target_inds).sum().float().item()
                union = torch.logical_or(pred_inds, target_inds).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def test_predict(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('/public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/best_model_DINOv2.pth', map_location=device))
    model.eval()
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(torch.unsqueeze(image, 1))
        output = nn.Softmax(dim=1)(output)
    return output.permute(0, 2, 3, 1).cpu().numpy()  # 移动到 CPU 并转换为 NumPy

# --------------------- 2. 数据处理和加载 --------------------- #
def load_data():
    prefix = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache/covid-segmentation'
    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)
    test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)
    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg

def onehot_to_mask(mask, palette):
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x.squeeze()

def split_data(images, masks, split_ratio=0.8):
    split_index = int(len(images) * split_ratio)
    train_images, val_images = images[:split_index], images[split_index:]
    train_masks, val_masks = masks[:split_index], masks[split_index:]
    return (train_images, train_masks), (val_images, val_masks)

class CovidSegDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        if self.transform:
            img, mask = self.transform(img, mask)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()
        # print('img_tensor.shape:',img_tensor.shape) img_tensor.shape: torch.Size([1, 512, 512])
        # print('mask_tensor.shape:',mask_tensor.shape) mask_tensor.shape: torch.Size([512, 512])
        return img_tensor, mask_tensor

def custom_transform(image, mask):
    # 随机水平翻转
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    
    # 随机垂直翻转
    if random.random() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    
    return image, mask
# --------------------- 3. 模型定义 --------------------- #
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, num_classes=4, in_channels=1):
#         super().__init__()
#         self.conv1 = DoubleConv(in_channels, 16)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = DoubleConv(16, 32)
#         self.pool2 = nn.MaxPool2d(2)
#         self.conv3 = DoubleConv(32, 64)
#         self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
#         self.conv4 = DoubleConv(64, 32)
#         self.up5 = nn.ConvTranspose2d(32, 16, 2, 2)
#         self.conv5 = DoubleConv(32, 16)
#         self.out = nn.Conv2d(16, num_classes, 1)
    
#     def forward(self, x):
#         c1 = self.conv1(x)
#         p1 = self.pool1(c1)
#         c2 = self.conv2(p1)
#         p2 = self.pool2(c2)
#         c3 = self.conv3(p2)
#         u4 = self.up4(c3)
#         c4 = self.conv4(torch.cat([u4, c2], dim=1))
#         u5 = self.up5(c4)
#         c5 = self.conv5(torch.cat([u5, c1], dim=1))
#         return self.out(c5)

# 空洞空间金字塔池化模块 (ASPP)
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

# 改进后的分割模型
class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes=4, patch_size=14, feat_dim=384):
        """
        基于 DINOv2 的改进分割模型
        """
        super(DinoSegmentationModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.feat_dim = feat_dim

        # 改进后的解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ASPP(256, 128),  # ASPP 模块
            nn.Conv2d(128, num_classes, kernel_size=1)  # 最终输出类别数
        )

    def preprocess(self, x):
        """
        输入数据预处理
        """
        transform = T.Compose([
            T.Resize((self.patch_size * 38, self.patch_size * 38)),
            T.Normalize(mean=(0.5,), std=(0.5,)),  # 医学图像通常使用单通道归一化
        ])
        x = torch.cat([x] * 3, dim=1)  # 扩展为 3 通道
        return transform(x)

    def forward(self, x):
        """
        前向传播
        """
        # 数据预处理
        x = self.preprocess(x / 255.0)

        # 提取特征
        features_dict = self.backbone.forward_features(x)
        features = features_dict['x_norm_patchtokens']

        # 重塑特征
        batch_size, num_patches, _ = features.shape
        patch_h = patch_w = int(num_patches ** 0.5)
        features = features.transpose(1, 2).reshape(batch_size, self.feat_dim, patch_h, patch_w)

        # 解码器
        output = self.decoder(features)
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return output

# --------------------- 4. 训练函数 --------------------- #
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, freeze_backbone=False):
    if freeze_backbone:
        # 冻结主干网络参数
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        # 解冻主干网络参数
        for param in model.backbone.parameters():
            param.requires_grad = True

    history = {
        "train_loss": [], "train_acc": [], "train_miou": [],
        "val_loss": [], "val_acc": [], "val_miou": []
    }
    best_val_miou = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_acc, train_miou = 0.0, 0.0, 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += pixel_accuracy(outputs, masks)
            train_miou += mIoU(outputs, masks, n_classes=4)
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_miou /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_acc, val_miou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_acc += pixel_accuracy(outputs, masks)
                val_miou += mIoU(outputs, masks, n_classes=4)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_miou /= len(val_loader)

        scheduler.step()

        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "best_model_DINOv2.pth")
            print(f"Epoch {epoch+1}: Best model saved with val_mIoU {best_val_miou:.4f}")

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_miou"].append(val_miou)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_miou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}")
    return history


# --------------------- 5. 主程序 --------------------- #
def main():
    # 加载数据
    images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()
    palette = [[0], [1], [2], [3]]
    masks_radiopedia_recover = np.array([onehot_to_mask(m, palette) for m in masks_radiopedia])
    masks_medseg_recover = np.array([onehot_to_mask(m, palette) for m in masks_medseg])
    (rad_train_imgs, rad_train_masks), (rad_val_imgs, rad_val_masks) = split_data(images_radiopedia, masks_radiopedia_recover, 0.8)
    (med_train_imgs, med_train_masks), (med_val_imgs, med_val_masks) = split_data(images_medseg, masks_medseg_recover, 0.8)
    train_imgs = np.concatenate((rad_train_imgs, med_train_imgs))
    train_masks = np.concatenate((rad_train_masks, med_train_masks))
    val_imgs = np.concatenate((rad_val_imgs, med_val_imgs))
    val_masks = np.concatenate((rad_val_masks, med_val_masks))

    train_dataset = CovidSegDataset(train_imgs, train_masks, transform=custom_transform)
    val_dataset = CovidSegDataset(val_imgs, val_masks, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = DinoSegmentationModel(backbone=dinov2_vitb14, num_classes=4, patch_size=14, feat_dim=384).to(device)

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=4e-4)  # 仅优化解码器参数
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    # 阶段1：训练解码器
    print("阶段1：训练解码器")
    history = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=15*7, device=device, freeze_backbone=True)

    # # 阶段2：整体训练
    # print("阶段2：整体训练模型")
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)  # 优化整个模型
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    # history = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=20, device=device, freeze_backbone=False)

    print("Training completed.")

    # 将 history 保存为 .pkl 文件
    with open('DINOv2_history.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    test_images_torch = torch.from_numpy(test_images_medseg).float()  # shape [10, 512, 512, 1]
    if len(test_images_torch.shape) == 4 and test_images_torch.shape[-1] == 1:
        test_images_torch = test_images_torch.permute(0, 3, 1, 2)  # [10, 1, 512, 512]
    # print('model(test_images_torch).shape:',model(test_images_torch).shape)
    output = np.zeros((10,512,512,4))
    for i in range(10):   
        output[i] = test_predict(model, test_images_torch[i])
    print('output.shape:',output.shape)
    
    test_masks_prediction = output > 0.5
    test_masks_prediction = test_masks_prediction[..., :-2]
    print('test_masks_prediction.shape:',test_masks_prediction.shape)

    submission = pd.DataFrame(
    data=np.stack(
        (np.arange(len(test_masks_prediction.ravel())),
         test_masks_prediction.ravel().astype(int)), axis=-1),
    columns=['Id', 'Predicted']).set_index('Id')

    submission.to_csv('submission.csv')
    print("训练与验证流程结束。")

if __name__ == "__main__":
    main()