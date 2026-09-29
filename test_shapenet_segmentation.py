# [SEGMENTATION ADDED FILE START: whole added file, lines 1-160]
import argparse
import json
import os

import torch

from pointnet import PointNetSegmenter
from shapenet_part_dataset import ShapeNetPartDataset
from train_shapenet_segmentation import (
    build_loader,
    evaluate_segmentation
)
import numpy as np

from show_shapenet_segmentation import visualize_sample


DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pointnet.pytorch",
        "data",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    )
)


DEFAULT_CHECKPOINT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "segmentation_output",
        "best_pointnet_shapenet_segmentation.pth"
    )
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VISUALIZATION_DIR = os.path.join(
    PROJECT_DIR,
    "segmentation_visualization",
    "test_first_5"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PointNet on ShapeNetPart segmentation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=2500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--out-json",
        type=str,
        default="segmentation_output/test_metrics.json"
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of test samples to visualize after evaluation"
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=DEFAULT_VISUALIZATION_DIR
    )
    parser.add_argument(
        "--visualization-top-k",
        type=int,
        default=5
    )
    parser.add_argument("--visualization-seed", type=int, default=42)
    return parser.parse_args()


def save_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Cannot find checkpoint: {}".format(checkpoint_path)
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    test_dataset = ShapeNetPartDataset(
        root=args.data_root,
        split="test",
        num_points=args.num_points,
        data_augmentation=False
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    num_parts = int(
        checkpoint.get(
            "num_parts",
            test_dataset.num_seg_classes
        )
    )
    if num_parts != test_dataset.num_seg_classes:
        raise ValueError(
            "Checkpoint has {} segmentation classes, dataset has {}".format(
                num_parts,
                test_dataset.num_seg_classes
            )
        )

    checkpoint_categories = checkpoint.get("category_names")
    if (
        checkpoint_categories is not None
        and list(checkpoint_categories) != test_dataset.category_names
    ):
        raise ValueError(
            "Checkpoint category order does not match the dataset"
        )

    feature_transform = bool(
        checkpoint.get("feature_transform", True)
    )

    model = PointNetSegmenter(
        num_parts=num_parts,
        feature_transform=feature_transform
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_segmentation(
        model,
        test_loader,
        test_dataset,
        device,
        max_batches=args.max_batches
    )
    metrics["split"] = "test"
    metrics["checkpoint"] = checkpoint_path
    metrics["best_val_miou"] = checkpoint.get("best_miou")

    save_json(args.out_json, metrics)

    # [SEGMENTATION] 测试完成后，使用同一个已加载模型保存前几个测试样本可视化。
    visualization_paths = []
    if args.num_visualizations > 0:
        num_samples = min(args.num_visualizations, len(test_dataset))
        np.random.seed(args.visualization_seed)
        print(
            "Generating visualizations for the first {} test samples...".format(
                num_samples
            )
        )

        for index in range(num_samples):
            output_path = os.path.join(
                args.visualization_dir,
                "test_sample_{:04d}.png".format(index)
            )
            saved_path, statistics = visualize_sample(
                model=model,
                dataset=test_dataset,
                index=index,
                split="test",
                output_path=output_path,
                top_k=args.visualization_top_k,
                device=device,
                show=False
            )
            visualization_paths.append(saved_path)
            print("Visualization saved:", saved_path)
            print(statistics)

    print("=" * 60)
    print("ShapeNetPart test finished")
    print("Test mIoU: {:.4f}".format(metrics["miou"]))
    for category, value in metrics["per_category_miou"].items():
        print("{:<12} {:.4f}".format(category, value))
    print("Metrics saved:", os.path.abspath(args.out_json))
    if visualization_paths:
        print(
            "Visualizations saved in:",
            os.path.abspath(args.visualization_dir)
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
# [SEGMENTATION ADDED FILE END: whole added file, lines 1-160]
