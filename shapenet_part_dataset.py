# [SEGMENTATION] ShapeNetPart data loader for the normal benchmark.
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# This file is added for ShapeNetPart segmentation.
# It adapts the "normal" benchmark layout, where every sample is stored as:
# x y z nx ny nz part_label
CATEGORY_PART_COUNTS = {
    "Airplane": 4,
    "Bag": 2,
    "Cap": 2,
    "Car": 4,
    "Chair": 4,
    "Earphone": 3,
    "Guitar": 3,
    "Knife": 2,
    "Lamp": 4,
    "Laptop": 2,
    "Motorbike": 6,
    "Mug": 2,
    "Pistol": 3,
    "Rocket": 3,
    "Skateboard": 3,
    "Table": 3,
}

SPLIT_FILES = {
    "train": "shuffled_train_file_list.json",
    "val": "shuffled_val_file_list.json",
    "test": "shuffled_test_file_list.json",
}


class ShapeNetPartDataset(Dataset):
    """Dataset for the ShapeNetPart normal benchmark."""

    def __init__(
        self,
        root,
        split="train",
        num_points=2500,
        data_augmentation=None,
        use_normals=False
    ):
        super(ShapeNetPartDataset, self).__init__()

        if split not in SPLIT_FILES:
            raise ValueError(
                "split must be one of: {}".format(
                    ", ".join(SPLIT_FILES.keys())
                )
            )

        if use_normals:
            raise NotImplementedError(
                "Using normals requires a 6-channel Input T-Net. "
                "The current PointNetSegmenter uses xyz only."
            )

        self.root = os.path.abspath(root)
        self.split = split
        self.num_points = int(num_points)
        self.data_augmentation = (
            split == "train"
            if data_augmentation is None
            else bool(data_augmentation)
        )
        self.use_normals = bool(use_normals)
        self.num_seg_classes = sum(CATEGORY_PART_COUNTS.values())

        self.category_file = os.path.join(
            self.root,
            "synsetoffset2category.txt"
        )
        if not os.path.isfile(self.category_file):
            raise FileNotFoundError(
                "Cannot find synsetoffset2category.txt: {}".format(
                    self.category_file
                )
            )

        self.synset_to_category = {}
        self.category_to_synset = {}
        with open(self.category_file, "r", encoding="utf-8") as file:
            for line in file:
                fields = line.strip().split()
                if len(fields) != 2:
                    continue
                category, synset_id = fields
                if category not in CATEGORY_PART_COUNTS:
                    raise ValueError(
                        "Unknown category in mapping file: {}".format(category)
                    )
                self.category_to_synset[category] = synset_id
                self.synset_to_category[synset_id] = category

        if set(self.category_to_synset) != set(CATEGORY_PART_COUNTS):
            raise ValueError(
                "Category list does not match CATEGORY_PART_COUNTS"
            )

        self.category_names = list(CATEGORY_PART_COUNTS.keys())
        self.category_to_index = {
            name: index
            for index, name in enumerate(self.category_names)
        }

        # Convert per-category local parts to the global 0-49 label space.
        offset = 0
        self.category_part_ids = {}
        for category in self.category_names:
            count = CATEGORY_PART_COUNTS[category]
            self.category_part_ids[category] = list(
                range(offset, offset + count)
            )
            offset += count

        split_file = os.path.join(
            self.root,
            "train_test_split",
            SPLIT_FILES[split]
        )
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                "Cannot find split file: {}".format(split_file)
            )

        with open(split_file, "r", encoding="utf-8") as file:
            file_list = json.load(file)

        self.samples = []
        for item in file_list:
            normalized_item = item.replace("\\", "/")
            parts = normalized_item.split("/")
            if len(parts) < 3:
                raise ValueError(
                    "Unexpected split item: {}".format(item)
                )

            synset_id = parts[-2]
            shape_id = parts[-1]
            if synset_id not in self.synset_to_category:
                raise ValueError(
                    "Unknown synset id in split file: {}".format(synset_id)
                )

            point_file = os.path.join(
                self.root,
                synset_id,
                shape_id + ".txt"
            )
            if not os.path.isfile(point_file):
                raise FileNotFoundError(
                    "Cannot find point file: {}".format(point_file)
                )

            category = self.synset_to_category[synset_id]
            category_index = self.category_to_index[category]
            self.samples.append(
                (point_file, category, category_index)
            )

        print(
            "ShapeNetPart {} samples: {}".format(split, len(self.samples))
        )
        print("Segmentation classes:", self.num_seg_classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_file, category, category_index = self.samples[index]
        raw = np.loadtxt(point_file, dtype=np.float32, ndmin=2)

        if raw.shape[1] < 4:
            raise ValueError(
                "Expected at least 4 columns in {}, got {}".format(
                    point_file,
                    raw.shape[1]
                )
            )

        points = raw[:, :3].copy()
        labels = raw[:, -1].astype(np.int64)

        # [SEGMENTATION] 对齐参考实现：始终有放回重采样到固定点数。
        choice = np.random.choice(
            len(labels),
            self.num_points,
            replace=True
        )

        points = points[choice]
        labels = labels[choice]

        center = np.mean(points, axis=0, keepdims=True)
        points = points - center

        scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        scale = max(float(scale), 1e-8)
        points = points / scale

        if self.data_augmentation:
            theta = np.random.uniform(0.0, np.pi * 2.0)
            rotation = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ],
                dtype=np.float32
            )
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation)
            points += np.random.normal(
                0.0,
                0.02,
                size=points.shape
            ).astype(np.float32)

        points = torch.from_numpy(points.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int64))
        category_index = torch.tensor(
            category_index,
            dtype=torch.long
        )

        # DataLoader output:
        # points [B, 3, N], labels [B, N], category index [B]
        points = points.transpose(0, 1).contiguous()

        return points, labels, category_index

    def get_category_part_ids(self, category_index):
        category = self.category_names[int(category_index)]
        return self.category_part_ids[category]


if __name__ == "__main__":
    default_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pointnet.pytorch",
            "data",
            "shapenetcore_partanno_segmentation_benchmark_v0_normal"
        )
    )

    dataset = ShapeNetPartDataset(
        root=default_root,
        split="train",
        num_points=256
    )
    points, labels, category_index = dataset[0]

    print("Points:", tuple(points.shape))
    print("Labels:", tuple(labels.shape))
    print("Category:", category_index.item())
    print("Label range:", labels.min().item(), labels.max().item())
