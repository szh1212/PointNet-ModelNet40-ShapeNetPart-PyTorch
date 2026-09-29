import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import S3DISDataset
from pointnet_seg import PointNetSegmentation


# ============================================================
# 1. 参数
# ============================================================

DATA_ROOT = (
    "./data/"
    "indoor3d_sem_seg_hdf5_data"
)

MODEL_PATH = (
    "./models/"
    "best_pointnet_s3dis.pth"
)

NUM_CLASSES = 13

BATCH_SIZE = 16

TEST_AREA = 5


CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter"
]


# ============================================================
# 2. device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("====================================")
print("PointNet S3DIS Test")
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

test_dataset = S3DISDataset(
    root_dir=DATA_ROOT,
    split="test",
    test_area=TEST_AREA
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Test samples:", len(test_dataset))
print("Test batches:", len(test_loader))


# ============================================================
# 4. Model
# ============================================================

model = PointNetSegmentation(
    num_classes=NUM_CLASSES,
    input_channels=9
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print()
print("Model loaded:")
print(MODEL_PATH)


# ============================================================
# 5. 统计数组
# ============================================================

total_correct = 0
total_seen = 0

class_correct = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

class_seen = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

class_intersection = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

class_union = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)


# ============================================================
# 6. Test
# ============================================================

with torch.no_grad():

    for batch_idx, (points, labels) in enumerate(
        test_loader
    ):

        points = points.to(device)

        labels = labels.to(device)

        # [B,4096,9]
        # ->
        # [B,9,4096]

        points = points.transpose(
            2,
            1
        )


        pred, _, _ = model(
            points
        )

        # [B,13,4096]
        # ->
        # [B,4096]

        predicted = torch.argmax(
            pred,
            dim=1
        )


        # ====================================================
        # Overall Accuracy
        # ====================================================

        total_correct += (
            predicted == labels
        ).sum().item()

        total_seen += labels.numel()


        pred_np = predicted.cpu().numpy()
        label_np = labels.cpu().numpy()


        # ====================================================
        # 每一类统计
        # ====================================================

        for cls in range(NUM_CLASSES):

            pred_mask = (
                pred_np == cls
            )

            label_mask = (
                label_np == cls
            )


            correct = np.sum(
                pred_mask & label_mask
            )

            seen = np.sum(
                label_mask
            )

            union = np.sum(
                pred_mask | label_mask
            )


            class_correct[cls] += correct

            class_seen[cls] += seen

            class_intersection[cls] += correct

            class_union[cls] += union


        if batch_idx % 100 == 0:

            print(
                f"Batch "
                f"[{batch_idx}/{len(test_loader)}]"
            )


# ============================================================
# 7. Overall Accuracy
# ============================================================

overall_accuracy = (
    total_correct
    /
    total_seen
)


# ============================================================
# 8. 每类 Accuracy
# ============================================================

class_accuracy = np.divide(
    class_correct,
    class_seen,
    out=np.zeros_like(class_correct),
    where=class_seen != 0
)


mean_class_accuracy = np.mean(
    class_accuracy[
        class_seen > 0
    ]
)


# ============================================================
# 9. IoU
# ============================================================

class_iou = np.divide(
    class_intersection,
    class_union,
    out=np.zeros_like(class_intersection),
    where=class_union != 0
)


mean_iou = np.mean(
    class_iou[
        class_union > 0
    ]
)


# ============================================================
# 10. 打印结果
# ============================================================

print()
print("====================================")
print("Final Results")
print("====================================")

print(
    f"Overall Accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Mean Class Accuracy: "
    f"{mean_class_accuracy * 100:.2f}%"
)

print(
    f"Mean IoU: "
    f"{mean_iou * 100:.2f}%"
)


print()
print("====================================")
print("Per-class IoU")
print("====================================")


for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:10s} "
        f"IoU: "
        f"{class_iou[i] * 100:.2f}%"
    )