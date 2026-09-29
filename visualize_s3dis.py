import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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
# 2. 每个类别一种颜色
# ============================================================

COLORS = np.array([
    [255,   0,   0],      # ceiling
    [  0, 255,   0],      # floor
    [  0,   0, 255],      # wall
    [255, 255,   0],      # beam
    [255,   0, 255],      # column
    [  0, 255, 255],      # window
    [255, 128,   0],      # door
    [128,   0, 255],      # table
    [  0, 128, 255],      # chair
    [255, 128, 128],      # sofa
    [128, 255, 128],      # bookcase
    [128, 128, 255],      # board
    [128, 128, 128]       # clutter
], dtype=np.float32) / 255.0


# ============================================================
# 3. device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("====================================")
print("S3DIS Visualization")
print("====================================")

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. Dataset
# ============================================================

test_dataset = S3DISDataset(
    root_dir=DATA_ROOT,
    split="test",
    test_area=TEST_AREA
)


# ============================================================
# 5. 随机取一个测试样本
# ============================================================

sample_index = random.randint(
    0,
    len(test_dataset) - 1
)

points, labels = test_dataset[
    sample_index
]


print()
print("Sample index:", sample_index)

print(
    "Points shape:",
    points.shape
)

print(
    "Labels shape:",
    labels.shape
)


# ============================================================
# 6. Model
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
# 7. 准备输入
# ============================================================

input_points = points.unsqueeze(0)

# [1,4096,9]

input_points = input_points.transpose(
    2,
    1
)

# [1,9,4096]

input_points = input_points.to(device)


# ============================================================
# 8. 推理
# ============================================================

with torch.no_grad():

    pred, _, _ = model(
        input_points
    )


# pred:
# [1,13,4096]

predicted = torch.argmax(
    pred,
    dim=1
)

# [1,4096]

predicted = predicted.squeeze(0)

# [4096]


predicted = predicted.cpu().numpy()

ground_truth = labels.numpy()


# ============================================================
# 9. 计算这个 block 的 accuracy
# ============================================================

accuracy = np.mean(
    predicted == ground_truth
)

print()
print(
    f"Sample Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 10. XYZ 坐标
# ============================================================

xyz = points[:, 0:3].numpy()


# ============================================================
# 11. Ground Truth 颜色
# ============================================================

gt_colors = COLORS[
    ground_truth
]


# ============================================================
# 12. Prediction 颜色
# ============================================================

pred_colors = COLORS[
    predicted
]


# ============================================================
# 13. 绘制 Ground Truth
# ============================================================

fig = plt.figure(
    figsize=(8, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    c=gt_colors,
    s=4
)

ax.set_title(
    "S3DIS Ground Truth"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.view_init(
    elev=30,
    azim=45
)

plt.tight_layout()

plt.savefig(
    "s3dis_ground_truth.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. 绘制 Prediction
# ============================================================

fig = plt.figure(
    figsize=(8, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    c=pred_colors,
    s=4
)

ax.set_title(
    f"S3DIS Prediction "
    f"(Acc: {accuracy * 100:.2f}%)"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.view_init(
    elev=30,
    azim=45
)

plt.tight_layout()

plt.savefig(
    "s3dis_prediction.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. 左右对比图
# ============================================================

fig = plt.figure(
    figsize=(16, 8)
)


# ------------------------------------------------------------
# 左边：Ground Truth
# ------------------------------------------------------------

ax1 = fig.add_subplot(
    121,
    projection="3d"
)

ax1.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    c=gt_colors,
    s=4
)

ax1.set_title(
    "Ground Truth"
)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax1.view_init(
    elev=30,
    azim=45
)


# ------------------------------------------------------------
# 右边：Prediction
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    122,
    projection="3d"
)

ax2.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    c=pred_colors,
    s=4
)

ax2.set_title(
    f"Prediction "
    f"(Acc: {accuracy * 100:.2f}%)"
)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

ax2.view_init(
    elev=30,
    azim=45
)


plt.tight_layout()

plt.savefig(
    "s3dis_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. 输出样本中出现的类别
# ============================================================

print()
print("Ground Truth classes:")

for cls in np.unique(
    ground_truth
):

    print(
        cls,
        CLASS_NAMES[cls]
    )


print()
print("Predicted classes:")

for cls in np.unique(
    predicted
):

    print(
        cls,
        CLASS_NAMES[cls]
    )


# ============================================================
# 17. 完成
# ============================================================

print()
print("====================================")
print("Visualization finished")
print("====================================")

print(
    "Saved:"
)

print(
    "s3dis_ground_truth.png"
)

print(
    "s3dis_prediction.png"
)

print(
    "s3dis_comparison.png"
)