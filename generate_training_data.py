import os
import cv2
import numpy as np
import glob
from tqdm import tqdm


def check_folder_structure(raw_folder, gt_folder):
    """检查文件夹结构"""
    print("=" * 50)
    print("文件夹结构检查")
    print("=" * 50)

    # 检查原始图像文件夹
    print(f"原始图像路径: {raw_folder}")
    print(f"文件夹存在: {os.path.exists(raw_folder)}")
    if os.path.exists(raw_folder):
        files = os.listdir(raw_folder)
        print(f"文件数量: {len(files)}")
        if files:
            print(f"前5个文件: {files[:5]}")
            extensions = set([os.path.splitext(f)[1].lower() for f in files])
            print(f"文件扩展名: {extensions}")
        else:
            print("文件夹为空！")
    else:
        print("文件夹不存在！")

    print("\n" + "-" * 50)

    # 检查GT图像文件夹
    print(f"GT图像路径: {gt_folder}")
    print(f"文件夹存在: {os.path.exists(gt_folder)}")
    if os.path.exists(gt_folder):
        files = os.listdir(gt_folder)
        print(f"文件数量: {len(files)}")
        if files:
            print(f"前5个文件: {files[:5]}")
            extensions = set([os.path.splitext(f)[1].lower() for f in files])
            print(f"文件扩展名: {extensions}")
        else:
            print("文件夹为空！")
    else:
        print("文件夹不存在！")

    print("=" * 50)


def simplest_color_balance(img, percent=1):
    """简化白平衡算法"""
    if len(img.shape) == 3:
        out_channels = []
        for channel in range(3):
            channel_img = img[:, :, channel]
            low_val = np.percentile(channel_img, percent)
            high_val = np.percentile(channel_img, 100 - percent)
            channel_img = np.clip(channel_img, low_val, high_val)
            channel_img = ((channel_img - low_val) / (high_val - low_val) * 255).astype(np.uint8)
            out_channels.append(channel_img)
        return np.stack(out_channels, axis=2)
    else:
        low_val = np.percentile(img, percent)
        high_val = np.percentile(img, 100 - percent)
        img = np.clip(img, low_val, high_val)
        img = ((img - low_val) / (high_val - low_val) * 255).astype(np.uint8)
        return img


def process_images_fixed(raw_folder, gt_folder, output_base='./'):
    """修复版处理函数"""

    # 1. 检查文件夹结构
    check_folder_structure(raw_folder, gt_folder)

    # 2. 获取所有支持的图像文件（不区分大小写）
    extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP']

    raw_images = []
    for ext in extensions:
        raw_images.extend(glob.glob(os.path.join(raw_folder, ext)))
    raw_images = sorted(raw_images)

    gt_images = []
    for ext in extensions:
        gt_images.extend(glob.glob(os.path.join(gt_folder, ext)))
    gt_images = sorted(gt_images)

    print(f"\n找到 {len(raw_images)} 张原始图像")
    print(f"找到 {len(gt_images)} 张GT图像")

    if len(raw_images) == 0:
        print("错误：未找到任何原始图像文件！")
        return

    if len(gt_images) == 0:
        print("错误：未找到任何GT图像文件！")
        return

    # 3. 创建输出目录
    output_dirs = ['input_train', 'input_wb_train', 'input_ce_train',
                   'input_gc_train', 'gt_train']

    for dir_name in output_dirs:
        os.makedirs(os.path.join(output_base, dir_name), exist_ok=True)
        print(f"创建目录: {os.path.join(output_base, dir_name)}")

    # 4. 处理图像
    processed_count = 0
    error_count = 0

    # 使用 enumerate 替代 range，更安全
    for i, raw_path in enumerate(tqdm(raw_images[:10000], desc="Processing images")):
        try:
            # 读取原始图像
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                print(f"警告：无法读取原始图像 {raw_path}")
                error_count += 1
                continue

            # 获取对应的GT图像（假设文件名相同）
            filename = os.path.basename(raw_path)
            gt_path = None

            # 尝试在GT文件夹中找到同名的文件（不同扩展名）
            base_name = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
                possible_gt = os.path.join(gt_folder, base_name + ext)
                if os.path.exists(possible_gt):
                    gt_path = possible_gt
                    break

            if gt_path is None:
                # 如果没找到同名文件，尝试按顺序匹配
                if i < len(gt_images):
                    gt_path = gt_images[i]
                else:
                    print(f"警告：找不到GT图像对应 {filename}")
                    error_count += 1
                    continue

            gt_img = cv2.imread(gt_path)
            if gt_img is None:
                print(f"警告：无法读取GT图像 {gt_path}")
                error_count += 1
                continue

            # 转换为RGB（用于处理）
            raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            # 1. 保存原始图像
            cv2.imwrite(os.path.join(output_base, 'input_train', filename), raw_img)

            # 2. 白平衡图像
            wb_img = simplest_color_balance(raw_img_rgb, percent=1)
            cv2.imwrite(os.path.join(output_base, 'input_wb_train', filename),
                        cv2.cvtColor(wb_img, cv2.COLOR_RGB2BGR))

            # 3. CLAHE 增强图像
            lab_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
            ce_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(output_base, 'input_ce_train', filename), ce_img)

            # 4. Gamma 校正图像
            gamma_img = np.power(raw_img.astype(np.float32) / 255.0, 0.7)
            gamma_img = (gamma_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_base, 'input_gc_train', filename), gamma_img)

            # 5. 保存GT图像
            cv2.imwrite(os.path.join(output_base, 'gt_train', filename), gt_img)

            processed_count += 1

        except Exception as e:
            print(f"处理图像 {raw_path} 时出错: {str(e)}")
            error_count += 1

    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 张")
    print(f"处理失败: {error_count} 张")


if __name__ == "__main__":
    # 配置路径 - 请根据实际情况修改
    raw_folder = "/Water-Net_Code-master/DATA_UIEB_mine/train/raw"
    gt_folder = "/Water-Net_Code-master/DATA_UIEB_mine/train/gt"

    # 验证路径是否正确
    print(f"当前工作目录: {os.getcwd()}")
    print(f"原始图像路径: {os.path.abspath(raw_folder)}")
    print(f"GT图像路径: {os.path.abspath(gt_folder)}")

    # 处理图像
    # 指定你想要的输出路径
    output_path = "/Water-Net_Code-master/data_UIEB/train"
    process_images_fixed(raw_folder, gt_folder, output_base=output_path)

# raw_folder填入训练集原始图片
# gt_folder填入训练集参考图片
# output_path填入想保存的路径

# 执行后会在指定路径文件夹下生成相应的input_ce_train、input_gc_train、input_wb_train、input_train、gt_train图片
