"""
批量计算UCIQE和UIQM
用法：修改image_dir的路径，然后运行
"""

import cv2
import numpy as np
import os
from pathlib import Path
import math
from skimage import transform
from scipy import ndimage


# ==================== UCIQE计算函数 ====================
def calculate_uciqe(image):
    """计算UCIQE (水下彩色图像质量评估)"""
    # image是BGR格式
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 根据论文的系数
    coe_metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_lab[..., 0] / 255.0
    img_a = img_lab[..., 1] / 255.0
    img_b = img_lab[..., 2] / 255.0

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))  # 色度
    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))  # 饱和度

    aver_sat = np.mean(img_sat)  # 平均饱和度
    aver_chr = np.mean(img_chr)  # 平均色度
    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))  # 色度方差

    # 亮度对比度
    nbins = 256
    hist, bins = np.histogram(img_lum * 255, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    # 最终质量值
    quality_val = coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat
    return quality_val


# ==================== UIQM子函数 ====================
def _uicm(img):
    """UICM: 水下图像色彩度测量"""
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B
    K = R.shape[0] * R.shape[1]

    # RG分量
    RG1 = RG.reshape(1, K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    # YB分量
    YB1 = YB.reshape(1, K)
    YB1 = np.sort(YB1)
    YB1 = YB1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)

    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaYB ** 2 + deltaRG ** 2)
    return uicm


def _uiconm(img):
    """UIConM: 水下图像对比度测量"""
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5

    # 调整图像大小使其能被patch整除
    m, n = R.shape
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        R = transform.resize(R, (x, y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))

    m, n = R.shape
    k1 = m / patchez
    k2 = n / patchez

    # 计算每个通道的AME
    def calculate_ame(channel):
        ame = 0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = channel[i:i + patchez, j:j + patchez]
                Max = np.max(patch)
                Min = np.min(patch)
                if (Max != 0 or Min != 0) and Max != Min:
                    ratio = (Max - Min) / (Max + Min)
                    ame += np.log(ratio) * ratio
        return abs(ame) / (k1 * k2)

    AMEER = calculate_ame(R)
    AMEEG = calculate_ame(G)
    AMEEB = calculate_ame(B)

    uiconm = AMEER + AMEEG + AMEEB
    return uiconm


def _uism(img):
    """UISM: 水下图像清晰度测量"""
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Sobel算子
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))

    patchez = 5
    m, n = R.shape

    # 调整大小
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))

    m, n = SobelR.shape
    k1 = m / patchez
    k2 = n / patchez

    # 计算每个通道的EME
    def calculate_eme(sobel_channel):
        eme = 0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = sobel_channel[i:i + patchez, j:j + patchez]
                Max = np.max(patch)
                Min = np.min(patch)
                if Max != 0 and Min != 0:
                    eme += np.log(Max / Min)
        return 2 * abs(eme) / (k1 * k2)

    EMER = calculate_eme(SobelR)
    EMEG = calculate_eme(SobelG)
    EMEB = calculate_eme(SobelB)

    # 权重系数
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism


def calculate_uiqm(img):
    """计算UIQM (水下图像质量测量)"""
    x = img.astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x)

    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm


# ==================== 批量处理函数 ====================
def batch_calculate_uciqe_uiqm(image_dir):
    """批量计算UCIQE和UIQM"""
    image_dir = Path(image_dir)

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 获取所有图像文件
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print("错误: 文件夹中没有找到图像文件！")
        return

    print(f"找到 {len(image_files)} 张图像")

    # 存储结果
    uciqe_values = []
    uiqm_values = []
    failed_images = []

    print("\n开始计算...")
    print("=" * 70)
    print(f"{'图像名称':<30} {'UCIQE':<15} {'UIQM':<15} {'状态':<10}")
    print("=" * 70)

    for img_file in image_files:
        try:
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                raise ValueError("无法读取图像")

            # 计算指标
            uciqe = calculate_uciqe(img)
            uiqm = calculate_uiqm(img)

            uciqe_values.append(uciqe)
            uiqm_values.append(uiqm)

            print(f"{img_file.name:<30} {uciqe:<15.4f} {uiqm:<15.4f} {'成功':<10}")

        except Exception as e:
            failed_images.append(img_file.name)
            print(f"{img_file.name:<30} {'N/A':<15} {'N/A':<15} {'失败':<10} ({str(e)})")

    # 输出统计结果
    print("=" * 70)
    print("\n计算结果统计:")
    print("-" * 40)
    print(f"总共处理: {len(image_files)} 张图像")
    print(f"成功处理: {len(uciqe_values)} 张图像")
    print(f"处理失败: {len(failed_images)} 张图像")

    if len(failed_images) > 0:
        print(f"失败的图像: {', '.join(failed_images[:10])}" + ("..." if len(failed_images) > 10 else ""))

    if uciqe_values:
        print("\nUCIQE统计:")
        print(f"  平均值: {np.mean(uciqe_values):.4f}")
        print(f"  最大值: {np.max(uciqe_values):.4f}")
        print(f"  最小值: {np.min(uciqe_values):.4f}")
        print(f"  标准差: {np.std(uciqe_values):.4f}")

        print("\nUIQM统计:")
        print(f"  平均值: {np.mean(uiqm_values):.4f}")
        print(f"  最大值: {np.max(uiqm_values):.4f}")
        print(f"  最小值: {np.min(uiqm_values):.4f}")
        print(f"  标准差: {np.std(uiqm_values):.4f}")

    return uciqe_values, uiqm_values


# ==================== 主程序 ====================
if __name__ == "__main__":
    # ============ 在这里设置你的图像路径 ============
    IMAGE_DIR = "/root/autodl-tmp/Water-Net_Code-master/result_new"  # 待评估图像文件夹路径

    # ============ 检查路径是否存在 ============
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 图像文件夹不存在: {IMAGE_DIR}")
        exit(1)

    # ============ 开始批量计算 ============
    print("批量计算UCIQE和UIQM")
    print("=" * 50)
    print(f"图像文件夹: {IMAGE_DIR}")
    print("=" * 50)

    uciqe_values, uiqm_values = batch_calculate_uciqe_uiqm(IMAGE_DIR)