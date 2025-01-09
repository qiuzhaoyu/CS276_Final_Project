import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd

# --------------------- 1. 指标函数 (pixel_accuracy & mIoU) --------------------- #
def pixel_accuracy(output, mask):
    """
    计算像素级精度 (pixel accuracy)
    Args:
        output: 模型输出, shape=[N, C, H, W]
        mask:   标签,   shape=[N, H, W] (单通道整数标签)
    """
    with torch.no_grad():
        # 先对输出做 softmax，然后取最大概率的下标（类别）
        output = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N, H, W]
        correct = (output == mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    """
    计算 mean IoU，适用于多分类分割
    Args:
        pred_mask: 模型输出, shape=[N, C, H, W]
        mask:      标签, shape=[N, H, W] (单通道整数标签)
        n_classes: 类别数 (含背景)
    """
    with torch.no_grad():
        # 1) softmax -> 取 argmax 作为预测类别
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]

        # 2) 打平为一维向量，方便统计
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(n_classes):
            pred_inds = (pred_mask == clas)
            target_inds = (mask == clas)
            if target_inds.long().sum().item() == 0:
                # 若该类在 GT 中没有像素，则跳过或记为 np.nan
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(pred_inds, target_inds).sum().float().item()
                union = torch.logical_or(pred_inds, target_inds).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)

def test_predict(model, image):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(torch.unsqueeze(image, 1))
        output = nn.Softmax(dim=1)(output)
    return output.permute(0, 2, 3, 1).cpu().numpy()  # 移动到 CPU 并转换为 NumPy


# --------------------- 2. 数据处理和加载 --------------------- #
prefix = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache/covid-segmentation'  # 修改为你的路径

def load_data():
    """加载训练和测试数据 (.npy 文件)，并转换为 numpy array"""
    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)
    test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)
    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg

def onehot_to_mask(mask, palette):
    """
    将 (H, W, K) one-hot 转为单通道 (H, W) 整数标签
    palette: [[0], [1], [2], [3]] 等
    """
    # 取 argmax 获得每个像素的类别索引
    x = np.argmax(mask, axis=-1)  # [H, W]
    # 根据 palette 映射（此处 palette 是 [[0],[1],[2],[3]]，实际上索引 == 值）
    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]  # shape (H, W, 1)
    return np.uint8(x).squeeze()          # 去掉通道 -> shape (H, W)

def split_data(images, masks, split_ratio=0.8):
    """
    按比例划分训练/验证集
    """
    split_index = int(len(images) * split_ratio)
    train_images, val_images = images[:split_index], images[split_index:]
    train_masks, val_masks = masks[:split_index], masks[split_index:]
    return (train_images, train_masks), (val_images, val_masks)

class CovidSegDataset(Dataset):
    """
    自定义 PyTorch Dataset，用于加载图像和 mask
    """
    def __init__(self, images, masks, transform=None):
        """
        Args:
            images: np.array, shape=[N, H, W, 1] (或 [N, H, W])，灰度图
            masks:  np.array, shape=[N, H, W], 单通道整数标签
            transform: 数据增广或预处理操作 (可自行扩展)
        """
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        
        # 如果图像是 (H, W, 1)，可以去掉最后一维
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)  # -> [H, W]
        
        # 归一化到 [0,1] (可根据数据实际情况选择)
        # img = img / 255.0

        # 可添加更多 transform，这里只做最简单的转换
        if self.transform is not None:
            # 自行实现或使用Albumentations等库
            img, mask = self.transform(img, mask)

        # 变成 tensor，注意通道顺序: (C, H, W)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.from_numpy(mask).long()              # [H, W]
        return img_tensor, mask_tensor


# --------------------- 3. 定义简易 U-Net 模型 (PyTorch) --------------------- #
class DoubleConv(nn.Module):
    """(Conv -> ReLU -> Conv -> ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=1):
        """
        num_classes: 分割类别数
        in_channels: 输入图像的通道数 (灰度=1, RGB=3)
        """
        super().__init__()
        # 下采样部分
        self.conv1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.conv3 = DoubleConv(32, 64)
        
        # 上采样部分
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(32, 16)

        # 最后的输出层
        self.out = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 编码器
        c1 = self.conv1(x)      # [16, H, W]
        p1 = self.pool1(c1)     # [16, H/2, W/2]

        c2 = self.conv2(p1)     # [32, H/2, W/2]
        p2 = self.pool2(c2)     # [32, H/4, W/4]

        # bottleneck
        c3 = self.conv3(p2)     # [64, H/4, W/4]

        # 解码器
        u4 = self.up4(c3)       # [32, H/2, W/2]
        cat4 = torch.cat([u4, c2], dim=1)  # 拼接
        c4 = self.conv4(cat4)   # [32, H/2, W/2]

        u5 = self.up5(c4)       # [16, H, W]
        cat5 = torch.cat([u5, c1], dim=1)
        c5 = self.conv5(cat5)   # [16, H, W]

        out = self.out(c5)      # [num_classes, H, W]
        return out


# --------------------- 4. 训练 & 验证循环 --------------------- #
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_miou = 0.0

    for imgs, masks in dataloader:
        imgs = imgs.to(device)     # [N, 1, H, W]
        masks = masks.to(device)   # [N, H, W]

        optimizer.zero_grad()
        outputs = model(imgs)      # [N, num_classes, H, W]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += pixel_accuracy(outputs, masks)
        epoch_miou += mIoU(outputs, masks, n_classes=4)

    # 计算平均值
    avg_loss = epoch_loss / len(dataloader)
    avg_acc = epoch_acc / len(dataloader)
    avg_miou = epoch_miou / len(dataloader)
    return avg_loss, avg_acc, avg_miou


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_miou = 0.0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            epoch_acc += pixel_accuracy(outputs, masks)
            epoch_miou += mIoU(outputs, masks, n_classes=4)

    avg_loss = epoch_loss / len(dataloader)
    avg_acc = epoch_acc / len(dataloader)
    avg_miou = epoch_miou / len(dataloader)
    return avg_loss, avg_acc, avg_miou

# --------------------- 5. 主程序：整合所有步骤 --------------------- #
def main():
    # 设置随机种子（可选）
    seed = 54
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) 加载数据 (numpy)
    images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()
    print("Radiopedia图像:", images_radiopedia.shape)
    print("Radiopedia掩膜:", masks_radiopedia.shape)
    print("MedSeg图像:", images_medseg.shape)
    print("MedSeg掩膜:", masks_medseg.shape)
    print("MedSeg测试图像:", test_images_medseg.shape)

    # 2) 将 one-hot 的掩膜转成单通道整数标签
    palette = [[0], [1], [2], [3]]
    masks_radiopedia_recover = np.array([onehot_to_mask(m, palette) for m in masks_radiopedia])
    masks_medseg_recover = np.array([onehot_to_mask(m, palette) for m in masks_medseg])

    # 3) 分别对 Radiopedia, MedSeg 数据集拆分 (train/val)
    (radiopedia_train_images, radiopedia_train_masks), (radiopedia_val_images, radiopedia_val_masks) = \
        split_data(images_radiopedia, masks_radiopedia_recover, split_ratio=0.9)

    (medseg_train_images, medseg_train_masks), (medseg_val_images, medseg_val_masks) = \
        split_data(images_medseg, masks_medseg_recover, split_ratio=0.9)

    # 4) 合并 train/val
    train_images = np.concatenate((radiopedia_train_images, medseg_train_images), axis=0)
    train_masks = np.concatenate((radiopedia_train_masks, medseg_train_masks), axis=0)
    val_images = np.concatenate((radiopedia_val_images, medseg_val_images), axis=0)
    val_masks = np.concatenate((radiopedia_val_masks, medseg_val_masks), axis=0)

    print("训练集大小:", len(train_images))
    print("验证集大小:", len(val_images))

    # 5) 构造 PyTorch Dataset 和 DataLoader
    train_dataset = CovidSegDataset(train_images, train_masks, transform=None)
    val_dataset = CovidSegDataset(val_images, val_masks, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 6) 定义模型、损失、优化器等
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = UNet(num_classes=4, in_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类分割
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 7) 训练循环
    epochs = 1
    for epoch in range(1, epochs+1):
        # ---- 训练 ----
        train_loss, train_acc, train_miou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # ---- 验证 ----
        val_loss, val_acc, val_miou = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, mIoU: {train_miou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, mIoU: {val_miou:.4f}")

    # 8) 测试推理 (如对 test_images_medseg 做分割预测)
    # 此处仅示例，如需评估需有对应的测试标签
    model.eval()
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

    submission.to_csv('sub.csv')
    print("训练与验证流程结束。")

if __name__ == "__main__":
    main()
