import torch
import torch.nn as nn
import torch.nn.functional as F


# CBAM Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out



# ASPP Module

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, dilation=dilations[0], bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=dilations[1], dilation=dilations[1], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=dilations[2], dilation=dilations[2], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=dilations[3], dilation=dilations[3], bias=False)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)


# Transformer Module (保持不变)
class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_feedforward=512, downsample_size=(32, 32)):
        super(TransformerBottleneck, self).__init__()
        self.in_channels = in_channels
        self.downsample_size = downsample_size  # 目标缩放尺寸，32x32 非常足够了

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x):
        # x shape: [B, C, H, W] (例如 256x256)
        b, c, h, w = x.shape

        # 1. Downsample (下采样)
        # 将 256x256 -> 32x32。序列长度从 65536 降为 1024。
        x_down = F.adaptive_avg_pool2d(x, self.downsample_size)

        # 2. Flatten for Transformer
        # [B, C, 32, 32] -> [B, C, 1024] -> [B, 1024, C]
        x_flatten = x_down.flatten(2).transpose(1, 2)

        # 3. Transformer Processing
        x_trans = self.transformer_encoder(x_flatten)

        # 4. Reshape back
        # [B, 1024, C] -> [B, C, 1024] -> [B, C, 32, 32]
        x_trans = x_trans.transpose(1, 2).reshape(b, c, self.downsample_size[0], self.downsample_size[1])

        # 5. Upsample (上采样) 恢复回原始尺寸
        # 使用双线性插值恢复到 [B, C, H, W]
        out = F.interpolate(x_trans, size=(h, w), mode='bilinear', align_corners=True)

        # 6. Residual Connection (残差连接)
        return x + out



# Improved WaterNet
class ImprovedWaterNet(nn.Module):
    def __init__(self):
        super(ImprovedWaterNet, self).__init__()

        # --- Main Branch (Confidence Map Generator) ---
        self.main_conv1 = nn.Conv2d(12, 128, 7, 1, 3)
        self.main_conv2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.main_conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.main_conv4 = nn.Conv2d(128, 64, 1, 1, 0)

        # [改进点1]: Transformer 用于捕获长距离依赖
        self.transformer = TransformerBottleneck(in_channels=64)

        # [改进点2]: CBAM 移至此处 (Feature Backend)
        # 此时输入是经过 Conv 和 Transformer 处理后的 64 通道深层特征
        # 它可以精确地对 "哪些特征决定了权重分配" 进行注意力加权
        self.cbam = CBAM(in_planes=64)

        self.main_conv5 = nn.Conv2d(64, 64, 7, 1, 3)
        self.main_conv6 = nn.Conv2d(64, 64, 5, 1, 2)
        self.main_conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.main_conv8 = nn.Conv2d(64, 3, 3, 1, 1)  # 输出最终权重 (3通道)

        # --- Refinement Branches (with ASPP) ---
        # WB Branch
        self.wb_conv1 = nn.Conv2d(6, 32, 7, 1, 3)
        self.wb_aspp = ASPP(32, 32)
        self.wb_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

        # CE Branch (对应原图的 HE/FTU)
        self.ce_conv1 = nn.Conv2d(6, 32, 7, 1, 3)
        self.ce_aspp = ASPP(32, 32)
        self.ce_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

        # GC Branch
        self.gc_conv1 = nn.Conv2d(6, 32, 7, 1, 3)
        self.gc_aspp = ASPP(32, 32)
        self.gc_conv3 = nn.Conv2d(32, 3, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, raw, wb, ce, gc):
        # 1. Main Branch: 生成权重图
        cat_main = torch.cat([raw, wb, ce, gc], dim=1)  # [B, 12, H, W]

        # 早期卷积特征提取
        m = self.relu(self.main_conv1(cat_main))
        m = self.relu(self.main_conv2(m))
        m = self.relu(self.main_conv3(m))
        m = self.relu(self.main_conv4(m))  # [B, 64, H, W]

        # [核心改进流]: Features -> Transformer -> CBAM -> Weights
        m = self.transformer(m)  # Global Context
        m = self.cbam(m)  # Feature Refinement (Attention)

        m = self.relu(self.main_conv5(m))
        m = self.relu(self.main_conv6(m))
        m = self.relu(self.main_conv7(m))

        weights = self.sigmoid(self.main_conv8(m))  # [B, 3, H, W]
        w_wb, w_ce, w_gc = torch.split(weights, 1, dim=1)

        # 2. Refinement Branches (with ASPP)
        # WB
        cat_wb = torch.cat([raw, wb], dim=1)
        r_wb = self.relu(self.wb_conv1(cat_wb))
        r_wb = self.wb_aspp(r_wb)
        r_wb = self.relu(self.wb_conv3(r_wb))

        # CE
        cat_ce = torch.cat([raw, ce], dim=1)
        r_ce = self.relu(self.ce_conv1(cat_ce))
        r_ce = self.ce_aspp(r_ce)
        r_ce = self.relu(self.ce_conv3(r_ce))

        # GC
        cat_gc = torch.cat([raw, gc], dim=1)
        r_gc = self.relu(self.gc_conv1(cat_gc))
        r_gc = self.gc_aspp(r_gc)
        r_gc = self.relu(self.gc_conv3(r_gc))

        # 3. Fusion
        out = r_wb * w_wb + r_ce * w_ce + r_gc * w_gc

        return out


if __name__ == '__main__':
    model = ImprovedWaterNet()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input, dummy_input, dummy_input, dummy_input)
    print("Output shape:", output.shape)
    print("CBAM successfully moved to backend.")