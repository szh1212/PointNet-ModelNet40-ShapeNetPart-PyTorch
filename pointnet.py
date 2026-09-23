import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. T-Net
# k=3  -> Input T-Net，学习 3×3 变换矩阵
# k=64 -> Feature T-Net，学习 64×64 变换矩阵
# ============================================================
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()

        self.k = k

        # ---------- MLP: k -> 64 -> 128 -> 1024 ----------
        # Conv1d(kernel_size=1) 相当于对每个点使用同一个 MLP
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # ---------- FC: 1024 -> 512 -> 256 -> k*k ----------
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 让初始变换矩阵接近单位矩阵
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):

        # x:
        # Input T-Net   -> [B, 3, N]
        # Feature T-Net -> [B, 64, N]

        batch_size = x.size(0)

        # [B, k, N] -> [B, 64, N]
        x = F.relu(self.bn1(self.conv1(x)))

        # [B, 64, N] -> [B, 128, N]
        x = F.relu(self.bn2(self.conv2(x)))

        # [B, 128, N] -> [B, 1024, N]
        x = F.relu(self.bn3(self.conv3(x)))

        # --------------------------------------------------
        # 对所有 N 个点做 max pooling
        # [B, 1024, N] -> [B, 1024]
        # --------------------------------------------------
        x = torch.max(x, 2)[0]

        # [B, 1024] -> [B, 512]
        x = F.relu(self.bn4(self.fc1(x)))

        # [B, 512] -> [B, 256]
        x = F.relu(self.bn5(self.fc2(x)))

        # [B, 256] -> [B, k*k]
        x = self.fc3(x)

        # --------------------------------------------------
        # 加单位矩阵 I
        # 初始状态让 T-Net 接近“不做变换”
        # --------------------------------------------------
        identity = torch.eye(
            self.k,
            device=x.device,
            dtype=x.dtype
        ).flatten().unsqueeze(0).repeat(batch_size, 1)

        x = x + identity

        # [B, k*k] -> [B, k, k]
        x = x.view(-1, self.k, self.k)

        return x


# ============================================================
# 2. PointNet Encoder
# 输入:
#       [B, 3, N]
#
# 输出:
#       global feature [B, 1024]
# ============================================================
class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()

        # Input T-Net
        self.input_tnet = TNet(k=3)

        # 第一阶段
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature T-Net
        self.feature_tnet = TNet(k=64)

        # 第二阶段
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):

        # ====================================================
        # x = [B, 3, N]
        # ====================================================

        # ---------- Input T-Net ----------
        input_transform = self.input_tnet(x)
        # [B, 3, 3]

        # bmm要求:
        # [B, N, 3] × [B, 3, 3]
        x = x.transpose(2, 1)
        # [B, N, 3]

        x = torch.bmm(x, input_transform)
        # [B, N, 3]

        x = x.transpose(2, 1)
        # [B, 3, N]

        # ---------- MLP 3 -> 64 -> 64 ----------
        x = F.relu(self.bn1(self.conv1(x)))
        # [B, 64, N]

        x = F.relu(self.bn2(self.conv2(x)))
        # [B, 64, N]

        # ---------- Feature T-Net ----------
        feature_transform = self.feature_tnet(x)
        # [B, 64, 64]

        x = x.transpose(2, 1)
        # [B, N, 64]

        x = torch.bmm(x, feature_transform)
        # [B, N, 64]

        x = x.transpose(2, 1)
        # [B, 64, N]

        # ---------- MLP 64 -> 64 -> 128 -> 1024 ----------
        x = F.relu(self.bn3(self.conv3(x)))
        # [B, 64, N]

        x = F.relu(self.bn4(self.conv4(x)))
        # [B, 128, N]

        x = self.bn5(self.conv5(x))
        # [B, 1024, N]

        # ---------- Symmetric Function: Max Pool ----------
        x = torch.max(x, 2)[0]
        # [B, 1024]

        return x, input_transform, feature_transform


# ============================================================
# 3. PointNet Classification Network
#
# global feature:
# 1024 -> 512 -> 256 -> num_classes
# ============================================================
class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNetClassifier, self).__init__()

        self.encoder = PointNetEncoder()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        # [B,3,N]
        x, input_transform, feature_transform = self.encoder(x)

        # [B,1024] -> [B,512]
        x = F.relu(self.bn1(self.fc1(x)))

        # [B,512] -> [B,256]
        x = F.relu(self.bn2(self.fc2(x)))

        x = self.dropout(x)

        # [B,256] -> [B,num_classes]
        x = self.fc3(x)

        return x, input_transform, feature_transform


# ============================================================
# 4. Feature Transform Regularization
#
# 希望 Feature T-Net 学出来的矩阵接近正交矩阵
#
# A * A^T ≈ I
# ============================================================
def feature_transform_regularizer(transform):

    batch_size = transform.size(0)
    k = transform.size(1)

    identity = torch.eye(
        k,
        device=transform.device,
        dtype=transform.dtype
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    product = torch.bmm(
        transform,
        transform.transpose(2, 1)
    )

    loss = torch.mean(
        torch.norm(product - identity, dim=(1, 2))
    )

    return loss


# ============================================================
# 5. 测试代码
# ============================================================
if __name__ == "__main__":

    # 自动选择 GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("====================================")
    print("PointNet Test")
    print("====================================")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # 假设:
    #
    # batch size = 8
    # 每个点云 = 1024 个点
    # 每个点 = xyz 三个坐标
    #
    # shape = [B, 3, N]
    #       = [8, 3, 1024]
    # --------------------------------------------------------

    B = 8
    N = 1024

    points = torch.randn(
        B,
        3,
        N,
        device=device
    )

    print("\nInput shape:")
    print(points.shape)

    # 创建 PointNet
    model = PointNetClassifier(
        num_classes=40
    ).to(device)

    # forward
    output, input_transform, feature_transform = model(points)

    print("\nOutput shape:")
    print(output.shape)

    print("\nInput Transform shape:")
    print(input_transform.shape)

    print("\nFeature Transform shape:")
    print(feature_transform.shape)

    # 正则项
    reg_loss = feature_transform_regularizer(
        feature_transform
    )

    print("\nFeature Transform Regularization:")
    print(reg_loss.item())

    # 参数量
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")
    print(total_params)

    print("\n====================================")
    print("PointNet forward SUCCESS!")
    print("====================================")