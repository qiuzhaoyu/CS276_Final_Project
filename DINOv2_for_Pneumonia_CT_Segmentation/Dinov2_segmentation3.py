import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置 TORCH_HOME
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

# --------------------- 2. 数据处理和加载 --------------------- #
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
        return img_tensor, mask_tensor

def custom_transform(image, mask):
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    if random.random() > 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    return image, mask

# --------------------- 3. 模型定义 --------------------- #
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
        x = torch.cat([x] * 3, dim=1)
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

# --------------------- 4. 训练函数 --------------------- #
def train_decoder(model, train_loader, optimizer, criterion, epochs, device):
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            masks = masks.argmax(dim=-1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Decoder Train Loss: {train_loss / len(train_loader):.4f}")

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler=None):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            masks = masks.argmax(dim=-1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

# --------------------- 5. 主程序 --------------------- #
def main():
    prefix = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache/covid-segmentation'
    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)

    train_dataset = CovidSegDataset(images_radiopedia, masks_radiopedia, transform=custom_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = DinoSegmentationModel(backbone=dinov2_vitb14, num_classes=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    print("Training decoder...")
    train_decoder(model, train_loader, optimizer, criterion, epochs=10, device="cuda")
    print("Fine-tuning entire model...")
    train_model(model, train_loader, None, optimizer, criterion, epochs=10, device="cuda")

if __name__ == "__main__":
    main()
