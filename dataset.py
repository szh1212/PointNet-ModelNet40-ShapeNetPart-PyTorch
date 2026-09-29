import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class S3DISDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        test_area=5
    ):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.test_area = test_area

        self.data = []
        self.label = []

        # ----------------------------------------------------
        # 1. 读取所有 h5 文件路径
        # ----------------------------------------------------
        all_files_path = os.path.join(
            root_dir,
            "all_files.txt"
        )

        with open(all_files_path, "r") as f:
            all_files = [
                line.strip()
                for line in f.readlines()
            ]

        # ----------------------------------------------------
        # 2. 读取每个 block 属于哪个房间
        # ----------------------------------------------------
        room_filelist_path = os.path.join(
            root_dir,
            "room_filelist.txt"
        )

        with open(room_filelist_path, "r") as f:
            room_list = [
                line.strip()
                for line in f.readlines()
            ]

        # ----------------------------------------------------
        # 3. 读入所有 h5
        # ----------------------------------------------------
        data_list = []
        label_list = []

        for file_name in all_files:

            # all_files.txt 里面可能带目录
            file_name = os.path.basename(file_name)

            file_path = os.path.join(
                root_dir,
                file_name
            )

            with h5py.File(file_path, "r") as h5_file:

                data = h5_file["data"][:]
                label = h5_file["label"][:]

                data_list.append(data)
                label_list.append(label)

        # ----------------------------------------------------
        # 4. 拼接全部数据
        # ----------------------------------------------------
        all_data = np.concatenate(
            data_list,
            axis=0
        )

        all_label = np.concatenate(
            label_list,
            axis=0
        )

        print("All data shape:", all_data.shape)
        print("All label shape:", all_label.shape)
        print("Room count:", len(room_list))

        # ----------------------------------------------------
        # 5. Area 5 划分训练 / 测试
        # ----------------------------------------------------
        indices = []

        target_area = "Area_" + str(test_area)

        for i, room_name in enumerate(room_list):

            if split == "train":

                # Area_5 不进入训练集
                if target_area not in room_name:
                    indices.append(i)

            elif split == "test":

                # 测试只使用 Area_5
                if target_area in room_name:
                    indices.append(i)

            else:
                raise ValueError(
                    "split must be 'train' or 'test'"
                )

        # ----------------------------------------------------
        # 6. 取出对应 split
        # ----------------------------------------------------
        self.data = all_data[indices]
        self.label = all_label[indices]

        print()
        print("====================================")
        print("S3DIS Dataset")
        print("====================================")
        print("Split:", split)
        print("Test Area:", test_area)
        print("Samples:", len(self.data))
        print("Data shape:", self.data.shape)
        print("Label shape:", self.label.shape)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        points = self.data[index]
        labels = self.label[index]

        # numpy -> torch
        points = torch.from_numpy(
            points
        ).float()

        labels = torch.from_numpy(
            labels.astype(np.int64)
        ).long()

        return points, labels


# ============================================================
# 测试 dataset.py
# ============================================================

if __name__ == "__main__":

    DATA_ROOT = (
        "./data/"
        "indoor3d_sem_seg_hdf5_data"
    )

    # --------------------------------------------------------
    # 训练集
    # --------------------------------------------------------
    train_dataset = S3DISDataset(
        root_dir=DATA_ROOT,
        split="train",
        test_area=5
    )

    # --------------------------------------------------------
    # 测试集
    # --------------------------------------------------------
    test_dataset = S3DISDataset(
        root_dir=DATA_ROOT,
        split="test",
        test_area=5
    )

    print()
    print("====================================")
    print("Check first training sample")
    print("====================================")

    points, labels = train_dataset[0]

    print("points shape:", points.shape)
    print("labels shape:", labels.shape)

    print()
    print("points dtype:", points.dtype)
    print("labels dtype:", labels.dtype)

    print()
    print("Unique labels:")
    print(torch.unique(labels))

    print()
    print("First point:")
    print(points[0])

    print()
    print("First 20 labels:")
    print(labels[:20])