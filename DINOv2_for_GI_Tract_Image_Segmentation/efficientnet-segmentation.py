import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
# from tqdm.notebook import tqdm  # 如果在脚本/终端，可改用 from tqdm import tqdm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
import numpy as np
import scipy
from scipy.ndimage import zoom
import pandas as pd
# 设置 TORCH_HOME
os.environ['TORCH_HOME'] = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache'
######################################
# 1. 数据加载
######################################
prefix = '/public_bme2/bme-dgshen/ZhaoyuQiu/.cache/covid-segmentation'  # 根据你的环境自行修改

def load_data():
    """加载训练和测试数据"""
    images_radiopedia = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
    masks_radiopedia = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
    images_medseg = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
    masks_medseg = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)
    test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)
    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg

images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images_medseg = load_data()

######################################
# 2. 数据处理（one-hot -> 单通道或RGB）
######################################
def onehot_to_mask(mask, palette):
    """
    将one-hot编码的mask转换为单通道(或RGB)格式
    Args:
        mask: shape (H, W, K)
        palette: 分类颜色编码，如 [[0], [1], [2], [3]]
    Returns:
        shape (H, W) 或 (H, W, C)
    """
    x = np.argmax(mask, axis=-1)  # [H, W]
    colour_codes = np.array(palette)  # 例如 [[0], [1], [2], [3]]
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

palette = [[0], [1], [2], [3]]
masks_radiopedia_recover = onehot_to_mask(masks_radiopedia, palette).squeeze() 
masks_medseg_recover = onehot_to_mask(masks_medseg, palette).squeeze()

######################################
# 3. 训练集与验证集划分
######################################

# 数据集划分
def split_data(images, masks, split_ratio=0.8):
    """
    将数据按比例划分为训练集和验证集
    Args:
        images: 图像数据 (numpy array)
        masks: 掩膜数据 (numpy array)
        split_ratio: 训练集比例
    Returns:
        train_images, train_masks, val_images, val_masks
    """
    split_index = int(len(images) * split_ratio)
    train_images, val_images = images[:split_index], images[split_index:]
    train_masks, val_masks = masks[:split_index], masks[split_index:]
    return train_images, train_masks, val_images, val_masks

# 对 MedSeg 和 Radiopedia 数据集分别进行划分
medseg_train_images, medseg_train_masks, medseg_val_images, medseg_val_masks = split_data(
    images_medseg, masks_medseg_recover, split_ratio=0.8
)

radiopedia_train_images, radiopedia_train_masks, radiopedia_val_images, radiopedia_val_masks = split_data(
    images_radiopedia, masks_radiopedia_recover, split_ratio=0.8
)

# 合并训练集
train_images = np.concatenate((medseg_train_images, radiopedia_train_images), axis=0)
train_masks = np.concatenate((medseg_train_masks, radiopedia_train_masks), axis=0)

# 合并验证集
val_images = np.concatenate((medseg_val_images, radiopedia_val_images), axis=0)
val_masks = np.concatenate((medseg_val_masks, radiopedia_val_masks), axis=0)

# 验证数据集大小
print(f"训练集大小: {len(train_images)} 张图像")
print(f"验证集大小: {len(val_images)} 张图像")
batch_size = 32 

######################################
# 4. 定义数据增广
######################################
TARGET_SIZE = 256

# 训练集图像增广
train_transform_img = T.Compose([
    T.RandomRotation(degrees=360, interpolation=T.InterpolationMode.NEAREST),
    T.RandomResizedCrop(
        size=(TARGET_SIZE, TARGET_SIZE),
        scale=(0.75, 1.0),
        ratio=(1.0, 1.0),
        interpolation=T.InterpolationMode.NEAREST
    ),
    T.RandomHorizontalFlip(p=0.5),
])

# 验证集图像增广（只缩放）
val_transform_img = T.Compose([
    T.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=T.InterpolationMode.NEAREST),
])

# 训练集掩码增广
train_transform_mask = T.Compose([
    T.RandomRotation(degrees=360, interpolation=T.InterpolationMode.NEAREST),
    T.RandomResizedCrop(
        size=(TARGET_SIZE, TARGET_SIZE),
        scale=(0.75, 1.0),
        ratio=(1.0, 1.0),
        interpolation=T.InterpolationMode.NEAREST
    ),
    T.RandomHorizontalFlip(p=0.5),
])

# 验证集掩码增广（只缩放）
val_transform_mask = T.Compose([
    T.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=T.InterpolationMode.NEAREST),
])

######################################
# 5. 自定义数据集类
######################################
class CovidDataset(Dataset):
    """COVID 数据集"""
    def __init__(
        self, images, masks,
        transform_img=None, transform_mask=None,
        is_train=True
    ):
        self.images = images
        self.masks = masks
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.is_train = is_train
        self.mean = [0.485]  # 根据数据分布可自行调整
        self.std = [0.229]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 取出 numpy 格式的图像和掩码
        image = self.images[idx]  # [H, W] 或 [H, W, 1]
        mask = self.masks[idx]    # [H, W]

        # 若图像是 (H, W, 1)，可 squeeze 到 (H, W)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        # 转为 PIL Image
        img_pil = Image.fromarray(image.astype(np.float32), mode='F')  # 32-bit 浮点灰度
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L')   # 8-bit 灰度

        # 数据增广：采用“同步随机种子”
        if self.is_train and self.transform_img is not None and self.transform_mask is not None:
            seed = np.random.randint(0, 999999)
            random.seed(seed)
            img_pil = self.transform_img(img_pil)

            random.seed(seed)
            mask_pil = self.transform_mask(mask_pil)
        else:
            # 验证集/测试集只执行简单transform
            if self.transform_img is not None:
                img_pil = self.transform_img(img_pil)
            if self.transform_mask is not None:
                mask_pil = self.transform_mask(mask_pil)

        # 转回 Tensor
        img_tensor = T.ToTensor()(img_pil)             
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

        # 标准化
        img_tensor = T.Normalize(self.mean, self.std)(img_tensor)

        return img_tensor, mask_tensor

######################################
# 6. 构建 DataLoader
######################################
train_dataset = CovidDataset(
    images=train_images,
    masks=train_masks,
    transform_img=train_transform_img,
    transform_mask=train_transform_mask,
    is_train=True
)

val_dataset = CovidDataset(
    images=val_images,
    masks=val_masks,
    transform_img=val_transform_img,
    transform_mask=val_transform_mask,
    is_train=False
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

######################################
# 7. 评价指标
######################################
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N, H, W]
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(n_classes):
            true_class = pred_mask == clas
            true_label = mask == clas
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

######################################
# 8. 模型与训练函数
######################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name='efficientnet-b2',
    in_channels=1,
    encoder_weights='imagenet',
    classes=4
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=20,   # 一个完整的余弦周期的epoch数
    eta_min=1e-5  # 最小学习率
)

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    """
    每个 Epoch 的验证阶段都根据 val_mIoU 判断是否是当前最好成绩，
    如果更好，则保存模型到 best_model.pth。
    同时返回包含训练过程数据的 history 字典。
    """
    model.to(device)

    best_val_miou = 0.0  # 用于记录目前为止的最佳mIoU
    best_epoch = -1

    # 用于存储训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_iou": [],
        "val_loss": [],
        "val_acc": [],
        "val_iou": [],
        "lr": []
    }

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            # print('images.shape:',images.shape)
            outputs = model(images)
            # print('outputs.shape:',outputs.shape)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += pixel_accuracy(outputs, masks)
            train_iou += mIoU(outputs, masks, n_classes=4)

        # 更新学习率
        scheduler.step()

        # 计算平均值
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = train_acc  / len(train_loader)
        avg_train_iou  = train_iou  / len(train_loader)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for v_images, v_masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                v_images, v_masks = v_images.to(device), v_masks.to(device)
                v_outputs = model(v_images)

                loss_v = criterion(v_outputs, v_masks)
                val_loss += loss_v.item()

                val_acc += pixel_accuracy(v_outputs, v_masks)
                val_iou += mIoU(v_outputs, v_masks, n_classes=4)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = val_acc  / len(val_loader)
        avg_val_iou  = val_iou  / len(val_loader)

        # ---------- 打印日志 ----------
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train | Loss: {avg_train_loss:.4f} | PixelAcc: {avg_train_acc:.4f} | mIoU: {avg_train_iou:.4f}")
        print(f"  Val   | Loss: {avg_val_loss:.4f} | PixelAcc: {avg_val_acc:.4f} | mIoU: {avg_val_iou:.4f}")

        # ---------- 保存最优模型 ----------
        if avg_val_iou > best_val_miou:
            best_val_miou = avg_val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  [*] Best val mIoU updated to {best_val_miou:.4f}. Model saved as best_model.pth.\n")
        else:
            print()

        # ---------- 记录历史 ----------
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["train_iou"].append(avg_train_iou)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)
        history["val_iou"].append(avg_val_iou)
        history["lr"].append(current_lr)

    print(f"训练结束，最佳模型出现在第 {best_epoch+1} 个 epoch，val mIoU = {best_val_miou:.4f}。")
    return history


######################################
# 9. 启动训练
######################################
history = fit(
    epochs=150,  # 可自行调整
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device
)

import pickle

# 将 history 保存为 .pkl 文件
with open('history.pkl', 'wb') as file:
    pickle.dump(history, file)



def test_predict(model, image, mean=[0.485], std=[0.229]):
    """
    预测单张图像的分割结果
    Args:
        model: 训练好的分割模型
        image: 输入的单张图像 (H, W, C)
        mean, std: 标准化参数
    Returns:
        output: 模型的原始输出 (N, H, W, C)
    """
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    # print(image.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('/public_bme2/bme-dgshen/ZhaoyuQiu/CS276_Final_Project/best_model.pth', map_location=device))
    model.eval()
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(torch.unsqueeze(image, 0))
        output = nn.Softmax(dim=1)(output)
    return output.permute(0, 2, 3, 1)

# 计算缩放因子
zoom_factors = (1, 256 / 512, 256 / 512, 1)
# 调整大小
image_batch = zoom(test_images_medseg, zoom_factors, order=1)  # order=1 表示线性插值
print("测试集图像形状:", image_batch.shape)



# 预测测试集的结果
output = np.zeros((10, 256, 256, 4))
for i in range(10):
    output[i] = test_predict(model, image_batch[i]).cpu().numpy()
print("输出形状:", output.shape)

# 生成提交文件
test_masks_prediction = output > 0.5
test_masks_prediction_original_size = scipy.ndimage.zoom(
    test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0
)

submission = pd.DataFrame(
    data=np.stack(
        (np.arange(len(test_masks_prediction_original_size.ravel())),
         test_masks_prediction_original_size.ravel().astype(int)), axis=-1
    ),
    columns=['Id', 'Predicted']
).set_index('Id')

submission.to_csv('sub.csv')