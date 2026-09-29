# [SEGMENTATION] ShapeNetPart prediction visualization.
import argparse
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from pointnet import PointNetSegmenter
from shapenet_part_dataset import ShapeNetPartDataset
from train_shapenet_segmentation import apply_part_mask


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
        description="Visualize ShapeNetPart ground truth and predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved PointNetSegmenter checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="segmentation_visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output PNG path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Matplotlib window after saving"
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Cannot find checkpoint: {}".format(path)
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False
    )


def build_model(checkpoint, default_num_parts):
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_parts = int(
            checkpoint.get("num_parts", default_num_parts)
        )
        feature_transform = bool(
            checkpoint.get("feature_transform", True)
        )
    else:
        state_dict = checkpoint
        num_parts = default_num_parts
        feature_transform = True

    model = PointNetSegmenter(
        num_parts=num_parts,
        feature_transform=feature_transform
    )
    model.load_state_dict(state_dict)
    return model


def set_axes_equal(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def format_statistics(
    category_name,
    num_points,
    error_count,
    error_rate,
    top_confusions
):
    lines = [
        "Category: {}".format(category_name),
        "Points: {}".format(num_points),
        "Error points: {}".format(error_count),
        "Error rate: {:.2f}%".format(error_rate),
        "",
        "Top confusion pairs:"
    ]

    if not top_confusions:
        lines.append("None")
    else:
        for (ground_truth, prediction), count in top_confusions:
            lines.append(
                "GT part {} -> Pred part {} : {}".format(
                    ground_truth,
                    prediction,
                    count
                )
            )

    return "\n".join(lines)


def visualize_sample(
    model,
    dataset,
    index,
    split,
    output_path,
    top_k=5,
    device=None,
    show=False
):
    if index < 0 or index >= len(dataset):
        raise IndexError(
            "index must be in [0, {}), got {}".format(
                len(dataset),
                index
            )
        )

    if device is None:
        device = next(model.parameters()).device

    points, labels, category_index = dataset[index]
    category_index_int = int(category_index.item())
    category_name = dataset.category_names[category_index_int]
    part_ids = dataset.get_category_part_ids(category_index_int)

    model.eval()
    points_batch = points.unsqueeze(0).to(device)
    category_batch = category_index.view(1).to(device)

    with torch.no_grad():
        logits, _, _ = model(points_batch)
        logits = apply_part_mask(
            logits,
            category_batch,
            dataset
        )
        predictions = logits.argmax(dim=-1)[0].cpu().numpy()

    labels_np = labels.cpu().numpy()
    xyz = points.transpose(0, 1).cpu().numpy()

    wrong_mask = predictions != labels_np
    error_count = int(wrong_mask.sum())
    error_rate = 100.0 * error_count / max(len(labels_np), 1)

    confusion_counter = Counter(
        zip(
            labels_np[wrong_mask].tolist(),
            predictions[wrong_mask].tolist()
        )
    )
    top_confusions = confusion_counter.most_common(top_k)

    part_to_local = {
        int(part_id): local_index
        for local_index, part_id in enumerate(part_ids)
    }
    palette = plt.get_cmap("tab10")
    max_parts = max(len(part_ids), 1)

    def colorize(part_labels):
        colors = np.zeros((len(part_labels), 4), dtype=np.float32)
        for point_index, part_label in enumerate(part_labels):
            local_index = part_to_local.get(int(part_label), 0)
            colors[point_index] = palette(
                local_index % max_parts
            )
        return colors

    colors_gt = colorize(labels_np)
    colors_pred = colorize(predictions)

    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        top=0.92,
        bottom=0.31,
        wspace=0.06
    )

    ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")

    ax_gt.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors_gt,
        s=3,
        depthshade=False
    )
    ax_pred.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors_pred,
        s=3,
        depthshade=False
    )

    if error_count > 0:
        ax_pred.scatter(
            xyz[wrong_mask, 0],
            xyz[wrong_mask, 1],
            xyz[wrong_mask, 2],
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            s=7,
            depthshade=False
        )

    for ax in (ax_gt, ax_pred):
        set_axes_equal(ax, xyz)
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()

    ax_gt.set_title("Ground Truth", fontsize=16)
    ax_pred.set_title("Prediction", fontsize=16)

    statistics = format_statistics(
        category_name,
        len(labels_np),
        error_count,
        error_rate,
        top_confusions
    )
    fig.text(
        0.05,
        0.04,
        statistics,
        ha="left",
        va="bottom",
        family="monospace",
        fontsize=11
    )

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160)

    if show:
        plt.show()

    plt.close(fig)
    return output_path, statistics


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(
        os.path.abspath(args.checkpoint),
        device
    )

    dataset = ShapeNetPartDataset(
        root=args.data_root,
        split=args.split,
        num_points=args.num_points,
        data_augmentation=False
    )

    model = build_model(
        checkpoint,
        dataset.num_seg_classes
    ).to(device)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        output_dir = os.path.abspath(args.out_dir)
        output_path = os.path.join(
            output_dir,
            "{}_sample_{:04d}.png".format(
                args.split,
                args.index
            )
        )

    output_path, statistics = visualize_sample(
        model=model,
        dataset=dataset,
        index=args.index,
        split=args.split,
        output_path=output_path,
        top_k=args.top_k,
        device=device,
        show=args.show
    )
    print("Visualization saved:", output_path)
    print(statistics)


if __name__ == "__main__":
    main()
