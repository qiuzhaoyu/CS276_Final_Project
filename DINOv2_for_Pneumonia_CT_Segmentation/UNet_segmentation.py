import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import pickle


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
    model.load_state_dict(torch.load('/public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/best_model.pth', map_location=device))
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
        return img_tensor, mask_tensor

# --------------------- 3. 模型定义 --------------------- #
class DoubleConv(nn.Module):
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
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv4 = DoubleConv(64, 32)
        self.up5 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv5 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, num_classes, 1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        u4 = self.up4(c3)
        c4 = self.conv4(torch.cat([u4, c2], dim=1))
        u5 = self.up5(c4)
        c5 = self.conv5(torch.cat([u5, c1], dim=1))
        return self.out(c5)

# --------------------- 4. 训练函数 --------------------- #
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
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

        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
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
    # 设置随机种子（可选）
    seed = 54
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()
    palette = [[0], [1], [2], [3]]
    masks_radiopedia_recover = np.array([onehot_to_mask(m, palette) for m in masks_radiopedia])
    masks_medseg_recover = np.array([onehot_to_mask(m, palette) for m in masks_medseg])
    (rad_train_imgs, rad_train_masks), (rad_val_imgs, rad_val_masks) = split_data(images_radiopedia, masks_radiopedia_recover, 0.9)
    (med_train_imgs, med_train_masks), (med_val_imgs, med_val_masks) = split_data(images_medseg, masks_medseg_recover, 0.9)
    train_imgs = np.concatenate((rad_train_imgs, med_train_imgs))
    train_masks = np.concatenate((rad_train_masks, med_train_masks))
    val_imgs = np.concatenate((rad_val_imgs, med_val_imgs))
    val_masks = np.concatenate((rad_val_masks, med_val_masks))
    print("训练集大小:", len(train_imgs))
    print("验证集大小:", len(val_imgs))
    train_dataset = CovidSegDataset(train_imgs, train_masks)
    val_dataset = CovidSegDataset(val_imgs, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5, device=device)
    print("Training completed.")

    # 将 history 保存为 .pkl 文件
    with open('history.pkl', 'wb') as file:
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

    submission.to_csv('sub.csv')
    print("训练与验证流程结束。")

if __name__ == "__main__":
    main()
