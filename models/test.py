import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model import ImprovedWaterNet


# --- 1. 自动预处理算法 (保持不变) ---
def white_balance(img):
    b, g, r = cv2.split(img)
    m, n = b.shape
    avg_b = np.mean(b); avg_g = np.mean(g); avg_r = np.mean(r)
    avg_b = 1 if avg_b == 0 else avg_b
    avg_g = 1 if avg_g == 0 else avg_g
    avg_r = 1 if avg_r == 0 else avg_r
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b; scale_g = avg_gray / avg_g; scale_r = avg_gray / avg_r
    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def histogram_equalization(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def gamma_correction(img, gamma=0.7):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


# --- [新增] Padding 函数 ---
def pad_to_multiple(x, multiple=16):
    """
    将图像 Tensor 长宽填充到 multiple 的倍数
    """
    h, w = x.shape[2], x.shape[3]
    new_h = (h // multiple + 1) * multiple if h % multiple != 0 else h
    new_w = (w // multiple + 1) * multiple if w % multiple != 0 else w
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    # F.pad 参数顺序: (左, 右, 上, 下)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x


# --- [修改] 预处理函数：加入 MAX_SIZE 缩放逻辑 ---
def preprocess_image(img_path, max_size=1024):
    raw_bgr = cv2.imread(img_path)
    if raw_bgr is None:
        print(f"Error: 无法读取 {img_path}")
        return None, None, None, None, None, 0, 0

    # === 1. 检查尺寸并进行安全缩放 ===
    h, w = raw_bgr.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > max_size:
        # 计算缩放比例
        scale = max_size / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)
        # 使用 OpenCV 进行缩放
        raw_bgr = cv2.resize(raw_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 更新当前长宽
        h, w = new_h, new_w

    # === 生成变体 (基于缩放后的图) ===
    wb_bgr = white_balance(raw_bgr)
    ce_bgr = histogram_equalization(raw_bgr)
    gc_bgr = gamma_correction(raw_bgr, gamma=0.7)

    # 转 RGB
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    wb_rgb = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2RGB)
    ce_rgb = cv2.cvtColor(ce_bgr, cv2.COLOR_BGR2RGB)
    gc_rgb = cv2.cvtColor(gc_bgr, cv2.COLOR_BGR2RGB)

    def to_tensor(img):
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    t_raw = to_tensor(raw_rgb)
    t_wb = to_tensor(wb_rgb)
    t_ce = to_tensor(ce_rgb)
    t_gc = to_tensor(gc_rgb)

    # 返回 Tensor 以及当前的 h, w (用于后续 Crop)
    return t_raw, t_wb, t_ce, t_gc, h, w


# --- 智能路径解析函数 (保持不变) ---
def resolve_input_path(input_arg):
    if os.path.isfile(input_arg): return [input_arg]
    candidates = [
        os.path.join(input_arg, "test", "test_real"),
        os.path.join(input_arg, "test_real"),
        os.path.join(input_arg, "test", "input_test"),
        input_arg
    ]
    target_dir = input_arg
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            files = os.listdir(p)
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')) for f in files):
                target_dir = p
                break
    print(f"Auto-detected image directory: {target_dir}")
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir)) if f.lower().endswith(extensions)]


# --- 2. 主推断逻辑 ---
def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 打印当前测试的模型配置 ---
    print("="*40)
    print(f"Inference Configuration:")
    print(f"  Transformer : {not args.no_transformer}")
    print(f"  CBAM        : {not args.no_cbam}")
    print(f"  ASPP        : {not args.no_aspp}")
    print(f"  Checkpoint  : {args.ckpt_path}")
    print("="*40)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- [关键修改] 实例化模型时传入开关参数 ---
    model = ImprovedWaterNet(
        use_transformer=not args.no_transformer,
        use_cbam=not args.no_cbam,
        use_aspp=not args.no_aspp
    ).to(device)
    
    print(f"Loading weights from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 使用 strict=True 可以确保你的开关参数和加载的权重文件完美匹配
    # 如果报错 key missing，说明你的命令行参数和训练时的配置不一致
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: Strict loading failed ({e}). Trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()

    image_paths = resolve_input_path(args.input_path)
    if len(image_paths) == 0:
        print(f"No images found in {args.input_path}.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # 定义 MAX_SIZE
    MAX_SIZE = 1024

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            fname = os.path.basename(img_path)

            # 预处理 (内部已包含 Resize 逻辑)
            raw, wb, ce, gc, h, w = preprocess_image(img_path, max_size=MAX_SIZE)
            
            if raw is None: continue

            raw = raw.to(device)
            wb = wb.to(device)
            ce = ce.to(device)
            gc = gc.to(device)

            # === 2. 自动 Padding (适配网络下采样) ===
            raw_padded = pad_to_multiple(raw, multiple=16)
            wb_padded = pad_to_multiple(wb, multiple=16)
            ce_padded = pad_to_multiple(ce, multiple=16)
            gc_padded = pad_to_multiple(gc, multiple=16)

            # 推理
            output = model(raw_padded, wb_padded, ce_padded, gc_padded)

            # === 3. 裁剪回有效尺寸 (Crop) ===
            # 注意：这里的 output 尺寸是缩放后的尺寸 (如果触发了 MAX_SIZE)
            output = output[:, :, :h, :w]

            # 后处理
            out_tensor = output.squeeze().permute(1, 2, 0).cpu().numpy()
            out_img = np.clip(out_tensor * 255.0, 0, 255).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(args.output_dir, fname)
            cv2.imwrite(save_path, out_img)

            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(image_paths)}] Processed: {fname} | Size: {w}x{h}")

    print(f"Done! Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImprovedWaterNet Inference')
    parser.add_argument('--input_path', type=str, default='DATA_UIEB_mine/test/raw', help='Dataset root or path to images')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/NoTrans_best.pth', help='Path to .pth file')
    parser.add_argument('--output_dir', type=str, default='result_NoTrans_UIEB', help='Folder to save results')
    
    parser.add_argument('--no_transformer', action='store_true', help='Disable Transformer Module')
    parser.add_argument('--no_cbam', action='store_true', help='Disable CBAM Module')
    parser.add_argument('--no_aspp', action='store_true', help='Disable ASPP Module')
    
    args = parser.parse_args()
    
    run_inference(args)
