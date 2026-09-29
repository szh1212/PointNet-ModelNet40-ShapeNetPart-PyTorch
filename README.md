# PointNet Part Segmentation on ShapeNetPart

A PyTorch reimplementation of PointNet for 3D point cloud part segmentation on the ShapeNetPart dataset.

This directory contains code for data loading, model definition, training, testing, metric calculation, and visualization. The experiment uses point coordinates as input and predicts a part category for each point. Invalid part labels are masked according to the ShapeNetPart category prior.

## Results Summary

| Metric | Result |
| --- | ---: |
| Best validation mIoU | **0.8303** |
| Test shape-level mIoU | **0.8017** |
| Test set samples | **2874** |
| Training epochs | **25** |
| Points sampled per batch | **2500** |

> The current evaluation script computes shape-level mIoU. It does not calculate test loss or point-wise accuracy. Training logs provide training loss, while the test results file provides test mIoU and per-category mIoU.

## Task Definition

ShapeNetPart part segmentation is a point-wise classification task on point clouds:

- Input: `N` 3D points for each shape, with shape `[B, 3, N]`
- Output: logits for 50 global part categories for each point, with shape `[B, N, 50]`
- Object categories: 16
- Part categories: 50 global part categories
- Each object category only allows prediction of valid parts belonging to that category

For example, an Airplane contains 4 valid part categories, a Chair contains 4, and a Motorbike contains 6. After applying the category mask, the model output retains only the part categories allowed for the current object category.

## Dataset

The ShapeNetPart normal benchmark is used. Each data file contains the following fields on each line:

```text
x y z nx ny nz part_label
```

The current model only reads `x y z` and `part_label`; the surface normals `nx ny nz` are not used.

The dataset split is:

| Split | Number of samples |
| --- | ---: |
| Train | 12137 |
| Validation | 1870 |
| Test | 2874 |

Data preprocessing and augmentation:

- Each sample is resampled with replacement to a fixed 2500 points
- Point cloud coordinates are centered and scaled to fit within a unit sphere
- Training samples are randomly rotated around the y-axis
- Gaussian noise with standard deviation 0.02 is added to the training set
- No data augmentation is applied to the validation or test sets
- The current input has 3 channels; normals are disabled because the existing Input T-Net is `3 x 3`

The dataset is large and is not included in the Git repository. Before running the code, place the dataset under the `segmentation` directory with the following structure:

```text
segmentation/
└── shapenetcore_partanno_segmentation_benchmark_v0_normal/
    ├── synsetoffset2category.txt
    ├── train_test_split/
    │   ├── shuffled_train_file_list.json
    │   ├── shuffled_val_file_list.json
    │   └── shuffled_test_file_list.json
    └── <synset_id>/
        └── <shape_id>.txt
```

## PyTorch Implementation

The core model definition is in `pointnet.py`, while data loading is implemented in `shapenet_part_dataset.py`.

The overall architecture is:

```text
Input Point Cloud
[B, 3, N]
        |
        v
Input T-Net
[B, 3, 3]
        |
        v
Shared MLP
3 -> 64 -> 64
        |
        v
Feature T-Net
[B, 64, 64]
        |
        +--------------------------+
        |                          |
        v                          v
Point Features [B, 64, N]     Shared MLP
                              64 -> 64 -> 128 -> 1024
                                       |
                                       v
                                  Max Pooling
                                       |
                                       v
                             Global Feature [B, 1024]
                                       |
                                       v
                         Repeat to N points [B, 1024, N]
                                       |
                                       v
                     Concatenate Point + Global Features
                              [B, 1088, N]
                                       |
                                       v
                              Segmentation MLP
                         1088 -> 512 -> 256 -> 128 -> 50
                                       |
                                       v
                              Per-point Predictions
                                   [B, N, 50]
```

### Input T-Net

The Input T-Net learns a `3 x 3` transformation matrix from the input point cloud to spatially align the input coordinates. The initial transformation output is constrained to be close to the identity matrix.

### Feature T-Net

The Feature T-Net learns a `64 x 64` transformation matrix from the 64-dimensional point-wise features produced by the first shared MLP. During training, an orthogonality regularization term is applied to encourage the transformation matrix to remain close to orthogonal.

### Shared MLP

The shared MLP is implemented using `Conv1d(kernel_size=1)`, which is equivalent to applying the same set of parameters independently to every point. The main encoder backbone is:

```text
3 -> 64 -> 64 -> 64 -> 128 -> 1024
```

The 64-dimensional point-wise features are retained for use by the subsequent segmentation head.

### Global and Point-wise Feature Concatenation

The encoder performs max pooling along the point dimension over the 1024-dimensional features to obtain a global shape feature `[B, 1024]`. This feature is repeated for every point and concatenated with the 64-dimensional point-wise features:

```text
Point feature:  [B, 64, N]
Global feature: [B, 1024, N]
Concatenated:   [B, 1088, N]
```

This allows the segmentation head to use both local point features and global shape context.

### Segmentation Head

The segmentation head uses a four-layer point-wise MLP:

```text
1088 -> 512 -> 256 -> 128 -> 50
```

The final output is transposed to `[B, N, 50]` and trained using CrossEntropyLoss for point-wise classification.

### Category Mask

ShapeNetPart uses a global part label space from 0 to 49, but each object category contains only a subset of the 50 part categories. During training and testing, the code constructs a valid-part mask according to the current object category and sets the logits of invalid part categories to `-1e9`. This prevents the model from predicting parts that do not belong to the current object category.

## Loss Function

The main loss is point-wise cross entropy:

```text
L_ce = CrossEntropyLoss(logits, part_labels)
```

When the Feature T-Net is enabled, an orthogonality regularization term is added:

```text
L_reg = mean(|| A * A^T - I ||_2)
L_total = L_ce + lambda * L_reg
```

The regularization weight used in the experiment is:

```text
lambda = 1e-3
```

## Training Configuration

| Configuration | Value |
| --- | ---: |
| Epochs | 25 |
| Batch size | 16 |
| Points per sample | 2500 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR scheduler | StepLR |
| Step size | 20 |
| Gamma | 0.5 |
| Feature transform | Enabled |
| Feature transform weight | 1e-3 |
| Random seed | 42 |
| Model selection | Best validation mIoU |

Validation mIoU is calculated after every training epoch. A new checkpoint is saved only when the validation mIoU exceeds the previous best value.

## Evaluation Metrics

The current implementation reports shape-level mean IoU.

For each shape, IoU is first calculated only for the valid part categories of the corresponding object category:

```text
IoU(part) = intersection(predicted_part, target_part)
            / union(predicted_part, target_part)
```

The IoUs of all valid parts within one shape are then averaged to obtain the shape-level IoU:

```text
shape_IoU = mean(IoU(part_1), ..., IoU(part_K))
```

Finally, the mean is taken over all test shapes:

```text
mIoU = mean(shape_IoU_1, ..., shape_IoU_S)
```

The implementation follows the reference evaluation procedure: if the union of the predicted and ground-truth points for a part is empty, the IoU for that part is set to 1.

## Experimental Results

The overall test results are saved in `segmentation_out/test_metrics.json`:

| Metric | Result |
| --- | ---: |
| Best validation mIoU | 0.8303 |
| Test shape-level mIoU | 0.8017 |
| Test shapes | 2874 |

### Training Log

The available epoch records in the terminal log start from epoch 12:

| Epoch | Train Loss | Validation mIoU |
| ---: | ---: | ---: |
| 12 | 0.2500 | 0.7997 |
| 13 | 0.2500 | 0.8004 |
| 14 | 0.2395 | 0.8043 |
| 15 | 0.2355 | 0.8028 |
| 16 | 0.2318 | 0.8132 |
| 17 | 0.2295 | 0.8052 |
| 18 | 0.2272 | 0.8138 |
| 19 | 0.2244 | 0.8162 |
| 20 | 0.2218 | 0.8219 |
| 21 | 0.2072 | 0.8295 |
| 22 | 0.2039 | **0.8303** |
| 23 | 0.2014 | 0.8184 |
| 24 | 0.1994 | 0.8219 |
| 25 | 0.1983 | 0.8266 |

Training loss generally decreases as the number of epochs increases. Validation mIoU reaches its best value of 0.8303 at epoch 22.

### Per-Category Test Results

| Category | Test mIoU |
| --- | ---: |
| Airplane | 0.7682 |
| Bag | 0.7416 |
| Cap | 0.7228 |
| Car | 0.6267 |
| Chair | 0.8752 |
| Earphone | 0.7068 |
| Guitar | 0.8774 |
| Knife | 0.8269 |
| Lamp | 0.7636 |
| Laptop | 0.9439 |
| Motorbike | 0.5669 |
| Mug | 0.8463 |
| Pistol | 0.7518 |
| Rocket | 0.4624 |
| Skateboard | 0.6159 |
| Table | 0.7992 |

Laptop, Guitar, Chair, and Mug have relatively high results. Rocket, Motorbike, Skateboard, and Car have relatively lower results, indicating that objects with elongated or thin structures, small parts, high structural variation, or less distinct internal part boundaries remain more challenging to segment.

## Visualization Comparison

The visualization script generates two side-by-side 3D point clouds:

- Left: Ground Truth
- Right: Prediction
- Colors represent part categories
- Black outlines indicate incorrectly predicted points
- The bottom displays the object category, number of incorrect points, error rate, and the major confused part categories

### Chair

![Chair ground truth and prediction](./segmentation_visualization/test_sample_0000.png)

### Lamp

![Lamp ground truth and prediction](./segmentation_visualization/test_sample_0009.png)

### Pistol

![Pistol ground truth and prediction](./segmentation_visualization/test_sample_0015.png)

These examples include both relatively accurate predictions and localized errors. Errors are typically concentrated around part boundaries, structural connections, or visually similar adjacent parts.
