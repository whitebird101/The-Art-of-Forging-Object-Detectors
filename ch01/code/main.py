import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# 模块 1：网络结构
# ---------------------------------------------------------
class SimpleDetector(nn.Module):
    
    def __init__(self, num_classes=2):
        super().__init__() # 调用父类 nn.Module 的初始化方法
        # 输出通道数：类别预测(num_classes) + 边界框等属性(8)
        self.out_channels = num_classes + 8

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # [B, 16, H/2, W/2]
            nn.ReLU(),                                # 激活函数，增加非线性
            nn.Conv2d(16, 32, 3, stride=2, padding=1),# [B, 32, H/4, W/4]
            nn.ReLU(),                                # 激活函数
            nn.Conv2d(32, 64, 3, stride=2, padding=1),# [B, 64, H/8, W/8]
            nn.ReLU(),                                # 激活函数
        )
        self.head = nn.Conv2d(64, self.out_channels, kernel_size=1)
        
        
    def forward(self, x):
        # 前向传播
        x = self.backbone(x)
        x = self.head(x)
        return x

# ---------------------------------------------------------
# 模块 2：数据集
# ---------------------------------------------------------
class SimpleFakeDataset(Dataset):
    
    def __init__(self, num_samples=100, img_size=256, num_classes=2, num_targets=3):
        super().__init__() #调用父类的初始化方法
        self.images = []  # 用于存储生成的模拟图像
        self.targets = [] # 用于存储生成的模拟网格化标签

        #网格尺寸 grid_size，输入图像尺寸除以网络的下采样倍率
        grid_size = img_size // 8

        # 目标张量的通道数
        channels = num_classes + 8

        # 在循环生成指定数量的图像
        for _ in range(num_samples):
            #模拟读取一张独立的图片
            single_img = torch.randn(3, img_size, img_size)
            # 模拟经过处理后的目标标注信息张量
            single_target = torch.randn(channels, grid_size, grid_size)
            # 将生成的单张图像和对应的真实标注信息存入列表
            self.images.append(single_img)
            self.targets.append(single_target)

    def __len__(self):
        # 返回数据集的总长度
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引返回对应的 (图像张量, 网格化标签张量)
        return self.images[idx], self.targets[idx]


# ---------------------------------------------------------
# 模块 3：验证评估，一个独立的方法
# ---------------------------------------------------------
def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0.0    #定义累计损失，初始化为0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader)
    mock_map = 1.0 / (avg_loss + 1e-6)
    return avg_loss, mock_map


# ---------------------------------------------------------
# 模块 4：训练中枢
# ---------------------------------------------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的计算设备: {device}") # 在控制台打印当前使用的计算设备

    img_size = 256      # 设定网络输入的图像分辨率大小
    num_classes = 2     # 设定模型需要检测的物体类别总数
    batch_size = 4      # 设定 DataLoader 每次打包送入网络的样本数量
    num_epochs = 20      # 设定模型需要遍历整个训练数据集的轮数

    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = SimpleFakeDataset(num_samples=80, img_size=img_size, num_classes=num_classes)
    val_dataset = SimpleFakeDataset(num_samples=20, img_size=img_size, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleDetector(num_classes=num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_mock_map = 0.0
    print("-" * 50) # 打印分割线，使输出日志更加美观清晰
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device) #将当前批次的训练图像转移到计算设备上
            targets = targets.to(device) #将当前批次的标注信息转移到计算设备上
            optimizer.zero_grad() #清空优化器中记录的上一轮梯度
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss, val_mock_map = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| Val Loss: {val_avg_loss:.4f} "
              f"| Val mAP: {val_mock_map:.4f}")
        current_save_path = os.path.join(save_dir, f"last.pth")
        torch.save(model.state_dict(), current_save_path)#保存模型
		#根据当前 epoch 验证集上的评估指标，保存最优模型
        if val_mock_map > best_mock_map:#如果是到当前为止最优的指标
            best_mock_map = val_mock_map
            print(f"  --> 发现更优模型 (mAP: {best_mock_map:.4f})，已保存至 best.pth")
            best_save_path = os.path.join(save_dir, "best.pth")
            torch.save(model.state_dict(), best_save_path)#保存模型




#程序入口
if __name__ == "__main__":
    train_model()
