import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from modelnet40_dataset import ModelNet40Dataset
from pointnet import PointNetClassifier


# ============================================================
# 1. 参数
# ============================================================

DATA_ROOT = "data/modelnet40_ply_hdf5_2048"

MODEL_PATH = "best_pointnet_modelnet40.pth"

NUM_POINTS = 1024

BATCH_SIZE = 32

NUM_CLASSES = 40


# ============================================================
# 2. ModelNet40 40个类别
# ============================================================

CLASS_NAMES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox"
]


# ============================================================
# 3. 测试整体准确率
# ============================================================

def evaluate(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for points, labels in test_loader:

            points = points.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # forward
            outputs, _, _ = model(points)

            # [B,40] -> [B]
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy, correct, total


# ============================================================
# 4. 主函数
# ============================================================

def main():

    print("=" * 60)
    print("PointNet ModelNet40 Test")
    print("=" * 60)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # 5. 加载测试集
    # ========================================================

    print()
    print("Loading ModelNet40 test dataset...")

    test_dataset = ModelNet40Dataset(
        root=DATA_ROOT,
        train=False,
        num_points=NUM_POINTS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print()
    print(
        "Test samples:",
        len(test_dataset)
    )

    # ========================================================
    # 6. 创建 PointNet
    # ========================================================

    model = PointNetClassifier(
        num_classes=NUM_CLASSES
    ).to(device)

    # ========================================================
    # 7. 加载训练好的权重
    # ========================================================

    print()
    print("Loading model:")

    print(MODEL_PATH)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("Model loaded successfully!")

    # ========================================================
    # 8. 整体测试
    # ========================================================

    print()
    print("=" * 60)
    print("Testing...")
    print("=" * 60)

    accuracy, correct, total = evaluate(
        model,
        test_loader,
        device
    )

    print()
    print("Correct:", correct)
    print("Total:", total)

    print(
        f"Test Accuracy: {accuracy:.2f}%"
    )

    # ========================================================
    # 9. 随机抽取 6 个点云
    # ========================================================

    print()
    print("=" * 60)
    print("Random Predictions")
    print("=" * 60)

    random_indices = random.sample(
        range(len(test_dataset)),
        6
    )

    # 创建图片
    fig = plt.figure(
        figsize=(15, 9)
    )

    # ========================================================
    # 10. 对6个样本逐个预测
    # ========================================================

    for i, index in enumerate(random_indices):

        points, label = test_dataset[index]

        # points:
        # [3,1024]

        # 增加 batch 维度
        #
        # [3,1024]
        #      ↓
        # [1,3,1024]

        input_points = points.unsqueeze(0)

        input_points = input_points.to(device)

        # ----------------------------------------------------
        # 预测
        # ----------------------------------------------------

        with torch.no_grad():

            output, _, _ = model(
                input_points
            )

            prediction = output.argmax(
                dim=1
            ).item()

        true_label = label.item()

        true_name = CLASS_NAMES[
            true_label
        ]

        predicted_name = CLASS_NAMES[
            prediction
        ]

        # ----------------------------------------------------
        # 打印预测
        # ----------------------------------------------------

        if prediction == true_label:
            result = "CORRECT"
        else:
            result = "WRONG"

        print(
            f"Sample {i + 1}: "
            f"True = {true_name} ({true_label}), "
            f"Pred = {predicted_name} ({prediction}), "
            f"{result}"
        )

        # ====================================================
        # 11. 画3D点云
        # ====================================================

        ax = fig.add_subplot(
            2,
            3,
            i + 1,
            projection="3d"
        )

        # points:
        # [3,1024]
        #
        # points[0] = x
        # points[1] = y
        # points[2] = z

        x = points[0].numpy()
        y = points[1].numpy()
        z = points[2].numpy()

        ax.scatter(
            x,
            y,
            z,
            s=2
        )

        # ----------------------------------------------------
        # 标题
        # ----------------------------------------------------

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 让点云看起来比例更正常
        ax.set_box_aspect(
            (
                max(x) - min(x),
                max(y) - min(y),
                max(z) - min(z)
            )
        )

    # ========================================================
    # 12. 保存图片
    # ========================================================

    plt.tight_layout()

    output_image = (
        "modelnet40_predictions.png"
    )

    plt.savefig(
        output_image,
        dpi=150
    )

    print()
    print(
        "Prediction image saved:",
        output_image
    )

    print()
    print("=" * 60)
    print("Test Finished!")
    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()