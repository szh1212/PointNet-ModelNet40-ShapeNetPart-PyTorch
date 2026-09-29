import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import S3DISDataset
from pointnet_seg import (
    PointNetSegmentation,
    feature_transform_regularizer
)


# ============================================================
# 1. 参数
# ============================================================

DATA_ROOT = (
    "./data/"
    "indoor3d_sem_seg_hdf5_data"
)

NUM_CLASSES = 13

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 0.001

TEST_AREA = 5

REG_WEIGHT = 0.001


# ============================================================
# 2. device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("====================================")
print("PointNet S3DIS Training")
print("====================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. Dataset
# ============================================================

train_dataset = S3DISDataset(
    root_dir=DATA_ROOT,
    split="train",
    test_area=TEST_AREA
)

test_dataset = S3DISDataset(
    root_dir=DATA_ROOT,
    split="test",
    test_area=TEST_AREA
)


# ============================================================
# 4. DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# 5. Model
# ============================================================

model = PointNetSegmentation(
    num_classes=NUM_CLASSES,
    input_channels=9
).to(device)


# ============================================================
# 6. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 7. 学习率调度器
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)


# ============================================================
# 8. 训练
# ============================================================

best_accuracy = 0.0


for epoch in range(EPOCHS):

    # ========================================================
    # Train
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for batch_idx, (points, labels) in enumerate(
        train_loader
    ):

        # ----------------------------------------------------
        # points:
        # [B,4096,9]
        #
        # labels:
        # [B,4096]
        # ----------------------------------------------------

        points = points.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # Conv1d 要求：
        #
        # [B,C,N]
        #
        # 所以：
        #
        # [B,4096,9]
        # →
        # [B,9,4096]
        # ----------------------------------------------------

        points = points.transpose(
            2,
            1
        )


        # ----------------------------------------------------
        # 梯度清零
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # forward
        # ----------------------------------------------------

        pred, input_transform, feature_transform = model(
            points
        )

        # pred:
        # [B,13,4096]


        # ----------------------------------------------------
        # Cross Entropy
        #
        # pred:
        # [B,13,4096]
        #
        # labels:
        # [B,4096]
        # ----------------------------------------------------

        classification_loss = F.cross_entropy(
            pred,
            labels
        )


        # ----------------------------------------------------
        # Feature Transform 正则化
        # ----------------------------------------------------

        reg_loss = feature_transform_regularizer(
            feature_transform
        )


        loss = (
            classification_loss
            +
            REG_WEIGHT * reg_loss
        )


        # ----------------------------------------------------
        # backward
        # ----------------------------------------------------

        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # 统计 loss
        # ----------------------------------------------------

        train_loss += loss.item()


        # ----------------------------------------------------
        # 预测类别
        #
        # pred:
        # [B,13,N]
        #
        # argmax(dim=1)
        #
        # →
        # [B,N]
        # ----------------------------------------------------

        predicted = torch.argmax(
            pred,
            dim=1
        )


        train_correct += (
            predicted == labels
        ).sum().item()


        train_total += labels.numel()


        # ----------------------------------------------------
        # 打印训练进度
        # ----------------------------------------------------

        if batch_idx % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # train accuracy
    # ========================================================

    train_accuracy = (
        100.0
        *
        train_correct
        /
        train_total
    )


    average_train_loss = (
        train_loss
        /
        len(train_loader)
    )


    # ========================================================
    # Test
    # ========================================================

    model.eval()

    test_correct = 0

    test_total = 0

    test_loss = 0.0


    with torch.no_grad():

        for points, labels in test_loader:

            points = points.to(device)

            labels = labels.to(device)


            points = points.transpose(
                2,
                1
            )


            pred, _, feature_transform = model(
                points
            )


            classification_loss = F.cross_entropy(
                pred,
                labels
            )


            reg_loss = feature_transform_regularizer(
                feature_transform
            )


            loss = (
                classification_loss
                +
                REG_WEIGHT * reg_loss
            )


            test_loss += loss.item()


            predicted = torch.argmax(
                pred,
                dim=1
            )


            test_correct += (
                predicted == labels
            ).sum().item()


            test_total += labels.numel()


    # ========================================================
    # test accuracy
    # ========================================================

    test_accuracy = (
        100.0
        *
        test_correct
        /
        test_total
    )


    average_test_loss = (
        test_loss
        /
        len(test_loader)
    )


    # ========================================================
    # 输出 epoch 结果
    # ========================================================

    print()
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Test Loss: "
        f"{average_test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )


    # ========================================================
    # 保存最好模型
    # ========================================================

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            "./models/"
            "best_pointnet_s3dis.pth"
        )

        print(
            "Best model saved!"
        )

        print(
            f"Best Accuracy: "
            f"{best_accuracy:.2f}%"
        )


    print(
        "===================================="
    )


    # ========================================================
    # 更新学习率
    # ========================================================

    scheduler.step()


# ============================================================
# 9. Finished
# ============================================================

print()
print("Training finished!")

print(
    f"Best Test Accuracy: "
    f"{best_accuracy:.2f}%"
)

print(
    "Model saved to:"
)

print(
    "./models/"
    "best_pointnet_s3dis.pth"
)