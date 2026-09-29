import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. T-Net
# ============================================================

class TNet(nn.Module):

    def __init__(self, k=3):
        super(TNet, self).__init__()

        self.k = k

        # ----------------------------------------------------
        # 逐点特征提取
        # ----------------------------------------------------
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # ----------------------------------------------------
        # 根据全局特征预测 k × k 变换矩阵
        # ----------------------------------------------------
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 初始时让预测部分为 0
        # 后面再加单位矩阵
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)


    def forward(self, x):

        # x:
        # [B, k, N]

        batch_size = x.size(0)

        # ----------------------------------------------------
        # Point MLP
        # ----------------------------------------------------
        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        # x:
        # [B, 1024, N]

        # ----------------------------------------------------
        # Global Max Pooling
        # ----------------------------------------------------
        x = torch.max(
            x,
            dim=2
        )[0]

        # [B, 1024]

        # ----------------------------------------------------
        # 全连接层
        # ----------------------------------------------------
        x = F.relu(
            self.bn4(
                self.fc1(x)
            )
        )

        x = F.relu(
            self.bn5(
                self.fc2(x)
            )
        )

        x = self.fc3(x)

        # [B, k*k]

        # ----------------------------------------------------
        # 加单位矩阵
        # ----------------------------------------------------
        identity = torch.eye(
            self.k,
            device=x.device,
            dtype=x.dtype
        )

        identity = identity.view(
            1,
            self.k * self.k
        )

        identity = identity.repeat(
            batch_size,
            1
        )

        x = x + identity

        # ----------------------------------------------------
        # [B, k*k]
        # →
        # [B, k, k]
        # ----------------------------------------------------
        x = x.view(
            -1,
            self.k,
            self.k
        )

        return x


# ============================================================
# 2. PointNet Encoder
# ============================================================

class PointNetEncoder(nn.Module):

    def __init__(self, input_channels=9):
        super(PointNetEncoder, self).__init__()

        self.input_channels = input_channels

        # ----------------------------------------------------
        # 输入空间变换
        #
        # 只变换 XYZ
        # ----------------------------------------------------
        self.input_tnet = TNet(k=3)

        # ----------------------------------------------------
        # Point MLP
        # ----------------------------------------------------
        self.conv1 = nn.Conv1d(
            input_channels,
            64,
            1
        )

        self.conv2 = nn.Conv1d(
            64,
            64,
            1
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # ----------------------------------------------------
        # Feature Transform
        # ----------------------------------------------------
        self.feature_tnet = TNet(k=64)

        # ----------------------------------------------------
        # 后半部分 Point MLP
        # ----------------------------------------------------
        self.conv3 = nn.Conv1d(
            64,
            64,
            1
        )

        self.conv4 = nn.Conv1d(
            64,
            128,
            1
        )

        self.conv5 = nn.Conv1d(
            128,
            1024,
            1
        )

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)


    def forward(self, x):

        # ====================================================
        # 输入：
        #
        # [B, 9, N]
        #
        # 9 =
        # XYZ + RGB + normalized XYZ
        # ====================================================

        batch_size = x.size(0)
        num_points = x.size(2)


        # ====================================================
        # 1. 输入空间变换
        # ====================================================

        xyz = x[:, 0:3, :]

        # xyz:
        # [B, 3, N]

        input_transform = self.input_tnet(
            xyz
        )

        # [B, 3, 3]


        # ----------------------------------------------------
        # 为了矩阵乘法：
        #
        # [B,3,N]
        # →
        # [B,N,3]
        # ----------------------------------------------------

        xyz = xyz.transpose(
            2,
            1
        )

        xyz = torch.bmm(
            xyz,
            input_transform
        )

        xyz = xyz.transpose(
            2,
            1
        )

        # xyz:
        # [B,3,N]


        # ----------------------------------------------------
        # 剩下的 RGB + normalized XYZ 不做空间旋转
        # ----------------------------------------------------

        if self.input_channels > 3:

            extra_features = x[:, 3:, :]

            x = torch.cat(
                [
                    xyz,
                    extra_features
                ],
                dim=1
            )

        else:

            x = xyz


        # ====================================================
        # 2. 第一个 Point MLP
        # ====================================================

        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        # [B,64,N]

        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        # [B,64,N]


        # ====================================================
        # 3. Feature Transform
        # ====================================================

        feature_transform = self.feature_tnet(
            x
        )

        # [B,64,64]


        # ----------------------------------------------------
        # [B,64,N]
        # →
        # [B,N,64]
        # ----------------------------------------------------

        x = x.transpose(
            2,
            1
        )

        x = torch.bmm(
            x,
            feature_transform
        )

        x = x.transpose(
            2,
            1
        )

        # [B,64,N]


        # ====================================================
        # 4. 保存局部特征
        #
        # 分割非常重要
        # ====================================================

        point_features = x

        # [B,64,N]


        # ====================================================
        # 5. 后半部分 Point MLP
        # ====================================================

        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = F.relu(
            self.bn4(
                self.conv4(x)
            )
        )

        x = self.bn5(
            self.conv5(x)
        )

        # x:
        # [B,1024,N]


        # ====================================================
        # 6. Global Max Pooling
        # ====================================================

        global_feature = torch.max(
            x,
            dim=2
        )[0]

        # [B,1024]


        # ====================================================
        # 7. 把 global feature 复制给每一个点
        # ====================================================

        global_feature = global_feature.unsqueeze(
            2
        )

        # [B,1024,1]

        global_feature = global_feature.repeat(
            1,
            1,
            num_points
        )

        # [B,1024,N]


        # ====================================================
        # 8. 局部 + 全局
        #
        # 64 + 1024 = 1088
        # ====================================================

        x = torch.cat(
            [
                point_features,
                global_feature
            ],
            dim=1
        )

        # [B,1088,N]


        return (
            x,
            input_transform,
            feature_transform
        )


# ============================================================
# 3. PointNet Semantic Segmentation
# ============================================================

class PointNetSegmentation(nn.Module):

    def __init__(
        self,
        num_classes=13,
        input_channels=9
    ):
        super(PointNetSegmentation, self).__init__()

        self.encoder = PointNetEncoder(
            input_channels=input_channels
        )

        # ----------------------------------------------------
        # 每个点：
        #
        # 1088
        # ↓
        # 512
        # ↓
        # 256
        # ↓
        # 128
        # ↓
        # 13
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            1088,
            512,
            1
        )

        self.conv2 = nn.Conv1d(
            512,
            256,
            1
        )

        self.conv3 = nn.Conv1d(
            256,
            128,
            1
        )

        self.conv4 = nn.Conv1d(
            128,
            num_classes,
            1
        )

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(
            p=0.3
        )


    def forward(self, x):

        # 输入：
        # [B,9,N]

        x, input_transform, feature_transform = (
            self.encoder(x)
        )

        # x:
        # [B,1088,N]


        x = F.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        # [B,512,N]


        x = F.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        # [B,256,N]


        x = F.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        # [B,128,N]


        x = self.dropout(x)


        x = self.conv4(x)

        # ================================================
        # 最终：
        #
        # [B,13,N]
        #
        # 每个点都有 13 个类别分数
        # ================================================

        return (
            x,
            input_transform,
            feature_transform
        )


# ============================================================
# 4. Feature Transform Regularization
# ============================================================

def feature_transform_regularizer(transform):

    # transform:
    # [B,64,64]

    batch_size = transform.size(0)

    k = transform.size(1)


    identity = torch.eye(
        k,
        device=transform.device
    )

    identity = identity.unsqueeze(0)

    identity = identity.repeat(
        batch_size,
        1,
        1
    )


    product = torch.bmm(
        transform,
        transform.transpose(2, 1)
    )


    loss = torch.mean(
        torch.norm(
            product - identity,
            dim=(1, 2)
        )
    )

    return loss


# ============================================================
# 5. 测试模型
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)


    model = PointNetSegmentation(
        num_classes=13,
        input_channels=9
    ).to(device)


    # 模拟：
    #
    # batch = 4
    # 9 个特征
    # 4096 个点

    x = torch.randn(
        4,
        9,
        4096
    ).to(device)


    output, input_transform, feature_transform = model(
        x
    )


    print()
    print("Input shape:")
    print(x.shape)


    print()
    print("Output shape:")
    print(output.shape)


    print()
    print("Input transform:")
    print(input_transform.shape)


    print()
    print("Feature transform:")
    print(feature_transform.shape)


    reg_loss = feature_transform_regularizer(
        feature_transform
    )


    print()
    print("Regularization loss:")
    print(reg_loss.item())