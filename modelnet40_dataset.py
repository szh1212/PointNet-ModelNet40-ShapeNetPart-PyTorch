import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40Dataset(Dataset):

    def __init__(
        self,
        root="data/modelnet40_ply_hdf5_2048",
        train=True,
        num_points=1024
    ):

        self.num_points = num_points

        # ====================================================
        # 1. 找到所有训练文件 / 测试文件
        # ====================================================

        if train:
            file_pattern = os.path.join(
                root,
                "ply_data_train*.h5"
            )
        else:
            file_pattern = os.path.join(
                root,
                "ply_data_test*.h5"
            )

        files = sorted(glob.glob(file_pattern))

        if len(files) == 0:
            raise FileNotFoundError(
                f"没有找到数据文件：{file_pattern}"
            )

        print("找到文件：")

        for file in files:
            print(" ", file)

        # ====================================================
        # 2. 读取所有 h5 文件
        # ====================================================

        all_data = []
        all_labels = []

        for file in files:

            with h5py.File(file, "r") as f:

                data = f["data"][:]
                labels = f["label"][:]

                all_data.append(data)
                all_labels.append(labels)

        # ====================================================
        # 3. 拼接
        # ====================================================

        self.data = np.concatenate(
            all_data,
            axis=0
        ).astype(np.float32)

        self.labels = np.concatenate(
            all_labels,
            axis=0
        ).reshape(-1).astype(np.int64)

        print()
        print("数据加载完成")
        print("data shape :", self.data.shape)
        print("label shape:", self.labels.shape)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # 原始:
        # [2048, 3]

        points = self.data[index]

        # 取前 num_points 个点
        # [1024, 3]
        points = points[:self.num_points]

        # 转成 PyTorch Tensor
        points = torch.from_numpy(points)

        # PointNet需要:
        #
        # [3, N]
        #
        # DataLoader组成batch后:
        #
        # [B, 3, N]

        points = points.transpose(0, 1)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return points, label


# ============================================================
# 测试 Dataset
# ============================================================

if __name__ == "__main__":

    print("==============================")
    print("测试训练集")
    print("==============================")

    train_dataset = ModelNet40Dataset(
        train=True,
        num_points=1024
    )

    print()
    print("训练样本数量:", len(train_dataset))

    points, label = train_dataset[0]

    print("单个点云 shape:", points.shape)
    print("标签:", label)

    print()
    print("==============================")
    print("测试测试集")
    print("==============================")

    test_dataset = ModelNet40Dataset(
        train=False,
        num_points=1024
    )

    print()
    print("测试样本数量:", len(test_dataset))

    points, label = test_dataset[0]

    print("单个点云 shape:", points.shape)
    print("标签:", label)