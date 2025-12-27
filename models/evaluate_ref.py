import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from PIL import Image
from tqdm import tqdm

# === 引入标准评估库 (必须安装: pip install scikit-image) ===
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
#              配置区域 (CONFIG)
# ==========================================
# 请在这里直接填写你的文件夹路径
# 注意：Windows路径如果报错，请把 \ 改为 /，或者在字符串前加 r
ENHANCED_DIR = r"D:\fuxian_model\Water-Net_Code-master\OutputImages"  # 增强后的图片文件夹 (你的结果图)
GT_DIR = r"D:\fuxian_model\Water-Net_Code-master\gt"  # 参考图片文件夹 (Ground Truth / 原图)


# ==========================================


# === 1. 专用评估 Dataset ===
class EvaluationDataset(torch.utils.data.Dataset):
    """
    用于加载增强图片和参考图片对的 Dataset。
    它假设增强图片文件夹和参考图片文件夹中的文件命名是相同的。
    """

    def __init__(self, enhanced_dir, gt_dir):
        self.enhanced_dir = enhanced_dir
        self.gt_dir = gt_dir

        # 验证目录是否存在
        if not os.path.isdir(self.enhanced_dir):
            raise FileNotFoundError(f"Error: 增强图路径不存在 -> {self.enhanced_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Error: 参考图路径不存在 -> {self.gt_dir}")

        # 筛选出在两个目录下都存在的图片名
        # 为了兼容不同的后缀大小写（比如 .jpg 和 .JPG），这里做一个简单的匹配逻辑
        enhanced_files = os.listdir(self.enhanced_dir)
        gt_files = os.listdir(self.gt_dir)

        # 这里假设文件名是包含后缀的完全匹配 (例如 "1.jpg" 对应 "1.jpg")
        # 如果你的文件名有前缀/后缀差异(比如 "1_UDCP.jpg" vs "1.jpg")，需要修改这里的逻辑
        # 针对你的 UDCP 代码，增强图可能有 "_UDCP" 后缀，我们尝试做一个智能匹配

        self.pairs = []  # 存储 (enhanced_name, gt_name)

        # 简单的名字匹配逻辑
        # 逻辑：遍历 GT 文件夹，去 Enhanced 文件夹里找对应的文件
        # 如果是 UDCP，输出文件通常是 原名前缀 + "_UDCP.jpg"

        for gt_name in gt_files:
            if not gt_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            prefix = gt_name.rsplit('.', 1)[0]  # 去掉后缀

            # 尝试找几种可能的增强图命名格式
            possible_names = [
                gt_name,  # 名字完全一样
                f"{prefix}_ULAP.jpg",  # UDCP 风格
                f"{prefix}_ULAP.png",
                f"{prefix}_enhanced.jpg"  # 其他常见风格
            ]

            found = False
            for try_name in possible_names:
                if try_name in enhanced_files:
                    self.pairs.append((try_name, gt_name))
                    found = True
                    break

            # 如果没找到，可以在这里打印一条 debug 信息
            # if not found:
            #     print(f"Warning: Missing enhanced image for {gt_name}")

        if not self.pairs:
            print(f"Error: 没有找到匹配的图片对。请检查文件夹路径和文件名是否对应。")
            print(f"Enhanced Dir: {self.enhanced_dir}")
            print(f"GT Dir: {self.gt_dir}")

        self.to_tensor = tfs.ToTensor()

    def __getitem__(self, index):
        enhanced_name, gt_name = self.pairs[index]

        enhanced_path = os.path.join(self.enhanced_dir, enhanced_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        # 打开图片
        enhanced_img = Image.open(enhanced_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # 警告：为确保度量计算正确，如果尺寸不一致，将增强图 resize 到参考图的尺寸
        if enhanced_img.size != gt_img.size:
            enhanced_img = enhanced_img.resize(gt_img.size, Image.Resampling.BILINEAR)

        # 转换为 Tensor (范围 [0, 1], 格式 [C, H, W])
        enhanced_tensor = self.to_tensor(enhanced_img)
        gt_tensor = self.to_tensor(gt_img)

        return enhanced_tensor, gt_tensor, enhanced_name

    def __len__(self):
        return len(self.pairs)


# === 2. 辅助函数 ===
def tensor_to_np(tensor):
    """
    将 PyTorch Tensor (B=1, C, H, W) 转换为 Numpy 格式 (H, W, C)，
    用于 Skimage 指标计算。范围默认为 [0, 1]。
    """
    # 确保张量在 CPU 上，并移除 batch 维度，C H W -> H W C
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


# === 3. 主评估函数 ===
def evaluate():
    # 使用配置区定义的路径
    print(f"Enhanced Dir: {ENHANCED_DIR}")
    print(f"GT Dir:       {GT_DIR}")

    try:
        # 数据集 (Batch Size 必须为 1)
        dataset = EvaluationDataset(ENHANCED_DIR, GT_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    if len(dataset) == 0:
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    print(f"==> Start Evaluation on {len(dataset)} image pairs...")

    with torch.no_grad():
        for enhanced, gt, name in tqdm(loader):
            # enhanced, gt 都是 [1, 3, H, W] 格式，范围 [0, 1]

            # 转换为 Numpy 格式计算指标
            enhanced_np = tensor_to_np(enhanced)
            gt_np = tensor_to_np(gt)

            # 计算 PSNR (data_range=1.0 表示输入值域为 [0, 1])
            try:
                cur_psnr = psnr_metric(gt_np, enhanced_np, data_range=1.0)
                # 计算 SSIM (channel_axis=2 表示通道维度在索引 2，即 H W C)
                # win_size 默认为 7，如果图片太小可能会报错，这里加个保险
                cur_ssim = ssim_metric(gt_np, enhanced_np, data_range=1.0, channel_axis=2)
            except ValueError as e:
                print(f"Skipping {name}: {e}")
                continue

            total_psnr += cur_psnr
            total_ssim += cur_ssim
            count += 1

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count

        print("\n" + "=" * 40)
        print("      Evaluation Results Summary      ")
        print(f" Model: UDCP (Assuming)")
        print(f" Evaluated Pairs: {count}")
        print(f" Avg PSNR: {avg_psnr:.4f} dB")
        print(f" Avg SSIM: {avg_ssim:.4f}")
        print("=" * 40)
    else:
        print("No valid images evaluated.")


if __name__ == '__main__':
    evaluate()