import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 我们刚刚写好的 ModelNet40 数据集
from modelnet40_dataset import ModelNet40Dataset

# 我们之前写好的 PointNet
from pointnet import PointNetClassifier, feature_transform_regularizer


# ============================================================
# 1. 参数设置
# ============================================================

BATCH_SIZE = 32

# 每个物体取 1024 个点
NUM_POINTS = 1024

# 先训练 20 轮
EPOCHS = 20

# 学习率
LEARNING_RATE = 0.001

# ModelNet40 一共 40 类
NUM_CLASSES = 40

# Feature T-Net 正则项系数
REG_WEIGHT = 0.001

# 数据集路径
DATA_ROOT = "data/modelnet40_ply_hdf5_2048"

# 最优模型保存路径
MODEL_SAVE_PATH = "best_pointnet_modelnet40.pth"


# ============================================================
# 2. 测试函数
# ============================================================

def evaluate(model, test_loader, device):

    # 切换到测试模式
    model.eval()

    correct = 0
    total = 0

    # 测试的时候不需要计算梯度
    with torch.no_grad():

        for points, labels in test_loader:

            # points:
            # [B, 3, 1024]

            points = points.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            outputs, _, _ = model(points)

            # outputs:
            # [B, 40]

            # 找概率最大的类别
            predictions = outputs.argmax(
                dim=1
            )

            # 统计正确数量
            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


# ============================================================
# 3. 主函数
# ============================================================

def main():

    print("=" * 60)
    print("PointNet - ModelNet40 Training")
    print("=" * 60)

    # ========================================================
    # 4. 选择 GPU
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # ========================================================
    # 5. 加载 ModelNet40
    # ========================================================

    print()
    print("=" * 60)
    print("Loading ModelNet40...")
    print("=" * 60)

    train_dataset = ModelNet40Dataset(
        root=DATA_ROOT,
        train=True,
        num_points=NUM_POINTS
    )

    test_dataset = ModelNet40Dataset(
        root=DATA_ROOT,
        train=False,
        num_points=NUM_POINTS
    )

    print()
    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # ========================================================
    # 6. DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        # 训练集一定打乱
        shuffle=True,

        # CPU并行读取数据
        num_workers=4,

        # GPU训练时加快 CPU -> GPU
        pin_memory=True,

        # 防止每个 epoch 都重新创建 worker
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True
    )

    # ========================================================
    # 7. 创建 PointNet
    # ========================================================

    model = PointNetClassifier(
        num_classes=NUM_CLASSES
    ).to(device)

    print()
    print("=" * 60)
    print("PointNet created")
    print("=" * 60)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Total parameters:",
        total_params
    )

    # ========================================================
    # 8. Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # 9. Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # 10. 学习率调度器
    #
    # 每 10 个 epoch：
    #
    # lr = lr × 0.5
    #
    # 0.001
    # ↓
    # 0.0005
    # ↓
    # 0.00025
    # ========================================================

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    # ========================================================
    # 11. 最优准确率
    # ========================================================

    best_accuracy = 0.0

    print()
    print("=" * 60)
    print("Start Training")
    print("=" * 60)

    # ========================================================
    # 12. Epoch
    # ========================================================

    for epoch in range(EPOCHS):

        start_time = time.time()

        # ----------------------------------------------------
        # 训练模式
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        # ====================================================
        # 13. 一个 epoch 中遍历所有 batch
        # ====================================================

        for batch_index, (
            points,
            labels
        ) in enumerate(train_loader):

            # points:
            #
            # [B, 3, 1024]

            points = points.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # =================================================
            # 第一步
            #
            # 清空上一个 batch 的梯度
            # =================================================

            optimizer.zero_grad()

            # =================================================
            # 第二步
            #
            # Forward
            # =================================================

            outputs, input_transform, feature_transform = model(
                points
            )

            # outputs:
            #
            # [B, 40]

            # =================================================
            # 第三步
            #
            # 分类损失
            # =================================================

            classification_loss = criterion(
                outputs,
                labels
            )

            # =================================================
            # 第四步
            #
            # Feature Transform Regularization
            #
            # 希望:
            #
            # A × A^T ≈ I
            # =================================================

            regularization_loss = (
                feature_transform_regularizer(
                    feature_transform
                )
            )

            # =================================================
            # 总 Loss
            # =================================================

            loss = (
                classification_loss
                +
                REG_WEIGHT
                * regularization_loss
            )

            # =================================================
            # 第五步
            #
            # Backward
            # =================================================

            loss.backward()

            # =================================================
            # 第六步
            #
            # 更新参数
            # =================================================

            optimizer.step()

            # =================================================
            # 统计 Loss
            # =================================================

            running_loss += (
                loss.item()
                * points.size(0)
            )

            # =================================================
            # 计算训练准确率
            # =================================================

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # =================================================
            # 每 50 个 batch 打印一次
            # =================================================

            if (
                (batch_index + 1) % 50 == 0
                or
                (batch_index + 1)
                == len(train_loader)
            ):

                print(
                    f"\r"
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch "
                    f"[{batch_index + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}",
                    end="",
                    flush=True
                )

        print()

        # ====================================================
        # 14. 一个 epoch 训练结束
        # ====================================================

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        train_accuracy = (
            100.0
            * correct
            / total
        )

        # ====================================================
        # 15. 测试集
        # ====================================================

        test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        # ====================================================
        # 16. 学习率
        # ====================================================

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # 17. 计算时间
        # ====================================================

        epoch_time = (
            time.time()
            - start_time
        )

        # ====================================================
        # 18. 打印结果
        # ====================================================

        print("-" * 60)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss:     "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Test Accuracy:  "
            f"{test_accuracy:.2f}%"
        )

        print(
            f"Learning Rate:  "
            f"{current_lr:.6f}"
        )

        print(
            f"Epoch Time:     "
            f"{epoch_time:.2f} s"
        )

        # ====================================================
        # 19. 保存最优模型
        # ====================================================

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                f"Best model saved! "
                f"Accuracy = "
                f"{best_accuracy:.2f}%"
            )

        print(
            f"Best Accuracy:  "
            f"{best_accuracy:.2f}%"
        )

        print("-" * 60)

        # ====================================================
        # 20. 更新学习率
        # ====================================================

        scheduler.step()

    # ========================================================
    # 21. 全部训练完成
    # ========================================================

    print()
    print("=" * 60)

    print("Training Finished!")

    print(
        f"Best Test Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        "Best Model:",
        MODEL_SAVE_PATH
    )

    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()