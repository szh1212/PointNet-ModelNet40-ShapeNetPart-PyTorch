# [SEGMENTATION] ShapeNetPart training entry point.
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet import (
    PointNetSegmenter,
    feature_transform_regularizer
)
from shapenet_part_dataset import ShapeNetPartDataset


DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pointnet.pytorch",
        "data",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointNet on ShapeNetPart segmentation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=2500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="segmentation_output")
    parser.add_argument(
        "--feature-transform",
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def build_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(dataset, **kwargs)


def build_part_mask(category_indices, dataset, device):
    batch_size = category_indices.size(0)
    num_parts = dataset.num_seg_classes
    mask = torch.zeros(
        batch_size,
        num_parts,
        dtype=torch.bool,
        device=device
    )

    for row, category_index in enumerate(category_indices.tolist()):
        part_ids = dataset.get_category_part_ids(category_index)
        mask[row, part_ids] = True

    return mask


def apply_part_mask(logits, category_indices, dataset):
    mask = build_part_mask(
        category_indices,
        dataset,
        logits.device
    )
    return logits.masked_fill(
        ~mask.unsqueeze(1),
        -1e9
    )


def evaluate_segmentation(
    model,
    loader,
    dataset,
    device,
    max_batches=None
):
    model.eval()

    shape_ious = []
    category_shape_ious = {
        name: []
        for name in dataset.category_names
    }

    with torch.no_grad():
        for batch_index, (points, labels, categories) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)

            logits, _, _ = model(points)
            logits = apply_part_mask(logits, categories, dataset)
            predictions = logits.argmax(dim=-1)

            for shape_index in range(labels.size(0)):
                category_index = int(categories[shape_index].item())
                category_name = dataset.category_names[category_index]
                part_ids = dataset.get_category_part_ids(category_index)

                part_ious = []
                for part_id in part_ids:
                    prediction_part = predictions[shape_index] == part_id
                    target_part = labels[shape_index] == part_id
                    intersection = (
                        prediction_part & target_part
                    ).sum().item()
                    union = (
                        prediction_part | target_part
                    ).sum().item()

                    # [SEGMENTATION] 对齐参考实现：并集为空时该部件 IoU 记为 1。
                    if union == 0:
                        iou = 1.0
                    else:
                        iou = intersection / float(union)
                    part_ious.append(iou)

                shape_iou = float(np.mean(part_ious))
                shape_ious.append(shape_iou)
                category_shape_ious[category_name].append(shape_iou)

    miou = float(np.mean(shape_ious)) if shape_ious else 0.0
    per_category_miou = {}
    for category_name, values in category_shape_ious.items():
        per_category_miou[category_name] = (
            float(np.mean(values)) if values else 0.0
        )

    return {
        "miou": miou,
        "per_category_miou": per_category_miou,
        "num_shapes": len(shape_ious)
    }


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset = ShapeNetPartDataset(
        root=args.data_root,
        split="train",
        num_points=args.num_points,
        data_augmentation=True
    )
    val_dataset = ShapeNetPartDataset(
        root=args.data_root,
        split="val",
        num_points=args.num_points,
        data_augmentation=False
    )
    test_dataset = ShapeNetPartDataset(
        root=args.data_root,
        split="test",
        num_points=args.num_points,
        data_augmentation=False
    )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = PointNetSegmenter(
        num_parts=train_dataset.num_seg_classes,
        feature_transform=args.feature_transform
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    best_miou = -1.0
    best_model_path = os.path.join(
        args.out_dir,
        "best_pointnet_shapenet_segmentation.pth"
    )

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        seen_batches = 0

        for batch_index, (
            points,
            labels,
            categories
        ) in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break

            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits, _, feature_transform = model(points)
            logits = apply_part_mask(
                logits,
                categories,
                train_dataset
            )

            loss = criterion(
                logits.reshape(-1, train_dataset.num_seg_classes),
                labels.reshape(-1)
            )

            if args.feature_transform:
                # [SEGMENTATION] 仅在启用 Feature T-Net 且返回矩阵时加入正则损失。
                if feature_transform is None:
                    raise RuntimeError(
                        "Feature transform is enabled, "
                        "but the model returned None."
                    )

                loss = loss + args.reg_weight * (
                    feature_transform_regularizer(feature_transform)
                )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            seen_batches += 1

            if (
                (batch_index + 1) % 20 == 0
                or batch_index == 0
            ):
                print(
                    "Epoch [{}/{}] Batch [{}/{}] Loss: {:.4f}".format(
                        epoch + 1,
                        args.epochs,
                        batch_index + 1,
                        len(train_loader),
                        loss.item()
                    )
                )

        train_loss = running_loss / max(seen_batches, 1)
        val_metrics = evaluate_segmentation(
            model,
            val_loader,
            val_dataset,
            device,
            max_batches=args.max_val_batches
        )

        print("-" * 60)
        print(
            "Epoch {}/{} Train Loss: {:.4f} Val mIoU: {:.4f}".format(
                epoch + 1,
                args.epochs,
                train_loss,
                val_metrics["miou"]
            )
        )
        print(
            "Epoch time: {:.2f} s".format(
                time.time() - start_time
            )
        )

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_miou": best_miou,
                    "num_parts": train_dataset.num_seg_classes,
                    "feature_transform": args.feature_transform,
                    "category_names": train_dataset.category_names,
                    "category_part_ids": train_dataset.category_part_ids,
                    "args": vars(args)
                },
                best_model_path
            )
            print("Best model saved:", best_model_path)

        scheduler.step()

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_segmentation(
        model,
        test_loader,
        test_dataset,
        device,
        max_batches=args.max_test_batches
    )
    test_metrics["best_val_miou"] = best_miou
    test_metrics["checkpoint"] = best_model_path

    save_json(
        os.path.join(args.out_dir, "test_metrics.json"),
        test_metrics
    )

    print("=" * 60)
    print("Segmentation training finished")
    print("Best validation mIoU: {:.4f}".format(best_miou))
    print("Test mIoU: {:.4f}".format(test_metrics["miou"]))
    for category, value in test_metrics["per_category_miou"].items():
        print("{:<12} {:.4f}".format(category, value))
    print("=" * 60)


if __name__ == "__main__":
    main()
