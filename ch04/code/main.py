import torch

from dataset import MultiscaleFakeDataset, build_dataloader
from model import YOLOLikeDetector
from trainer import SimpleTrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    img_size = 256
    num_classes = 2
    batch_size = 4
    num_epochs = 20

    # 1. 创建数据集
    train_dataset = MultiscaleFakeDataset(80, img_size, num_classes)
    val_dataset = MultiscaleFakeDataset(20, img_size, num_classes)

    # 2. 创建 DataLoader
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. 创建模型
    model = YOLOLikeDetector(num_classes=num_classes)

    # 4. 创建训练器
    trainer = SimpleTrainer(
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-3,
        save_dir="weights"
    )

    # 5. 开始训练
    trainer.train(num_epochs=num_epochs, img_size=img_size)


if __name__ == "__main__":
    main()