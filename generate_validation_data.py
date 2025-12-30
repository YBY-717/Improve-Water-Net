import os
import cv2
import numpy as np
import glob


def process_test_simple(raw_folder, gt_folder, output_base='./'):
    """
    简化版测试集处理
    :param raw_folder: 原始图像文件夹路径
    :param gt_folder: GT图像文件夹路径
    :param output_base: 指定输出的根目录 (默认为当前目录 './')
    """

    # 定义子文件夹名称
    sub_dirs = {
        'raw': 'input_val',
        'wb': 'input_wb_val',
        'ce': 'input_ce_val',
        'gc': 'input_gc_val',
        'gt': 'gt_val'
    }

    # 1. 创建输出目录 (拼接 output_base)
    for key, dir_name in sub_dirs.items():
        # 使用 os.path.join 拼接路径，确保跨平台兼容且路径正确
        full_dir_path = os.path.join(output_base, dir_name)
        os.makedirs(full_dir_path, exist_ok=True)
        print(f"创建目录: {full_dir_path}")

    # 获取所有图像文件
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    raw_images = []
    for ext in extensions:
        raw_images.extend(glob.glob(os.path.join(raw_folder, ext)))

    raw_images = sorted(raw_images)

    print(f"找到 {len(raw_images)} 张测试图像")

    # 处理每张图像
    for i, raw_path in enumerate(raw_images[:10000]):  # 最多10000张
        try:
            filename = os.path.basename(raw_path)
            # print(f"处理: {filename}") # 稍微减少打印量，或者保留

            # 读取图像
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                continue

            # 查找对应的GT图像
            base_name = os.path.splitext(filename)[0]
            gt_path = None

            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                possible_gt = os.path.join(gt_folder, base_name + ext)
                if os.path.exists(possible_gt):
                    gt_path = possible_gt
                    break

            if gt_path:
                gt_img = cv2.imread(gt_path)
            else:
                gt_img = np.zeros_like(raw_img)

            # --- 保存图像 (关键修改点：使用 os.path.join 拼接 output_base) ---

            # 1. 保存原始图像 -> output_base/input_val/filename
            cv2.imwrite(os.path.join(output_base, sub_dirs['raw'], filename), raw_img)

            # 2. 白平衡（简化）
            avg = raw_img.mean(axis=(0, 1))
            gray = avg.mean()
            scale = gray / (avg + 1e-6)
            wb_img = np.clip(raw_img * scale, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_base, sub_dirs['wb'], filename), wb_img)

            # 3. CLAHE增强
            lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            ce_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(output_base, sub_dirs['ce'], filename), ce_img)

            # 4. Gamma校正
            gamma_img = np.power(raw_img.astype(np.float32) / 255.0, 0.7)
            gamma_img = (gamma_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_base, sub_dirs['gc'], filename), gamma_img)

            # 5. 保存GT
            cv2.imwrite(os.path.join(output_base, sub_dirs['gt'], filename), gt_img)

        except Exception as e:
            print(f"错误: {e}")
            continue

    print(f"测试集处理完成！结果已保存至: {output_base}")


# 使用示例
if __name__ == "__main__":
    # 输入路径
    raw_folder = "/Water-Net_Code-master/DATA_UIEB_mine/val/raw"
    gt_folder = "/Water-Net_Code-master/DATA_UIEB_mine/val/gt"

    # --- 这里指定你想保存的路径 ---
    output_dir = "/Water-Net_Code-master/data_UIEB/val"

    # 调用函数时传入 output_dir
    process_test_simple(raw_folder, gt_folder, output_base=output_dir)

# raw_folder填入验证集原始图片
# gt_folder填入验证集参考图片
# output_dir填入想保存的路径

# 执行后会在指定路径文件夹下生成相应的input_ce_val、input_gc_val、input_wb_val、input_val、gt_val图片
