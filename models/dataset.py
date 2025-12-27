import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WaterDataset(Dataset):
    def __init__(self, dataset_dir, split='train', image_size=(256, 256)):
        """
        dataset_dir: 数据集根目录
        split: 'train' 或 'test'
        image_size: 强制缩放的目标尺寸 (height, width)
        """
        super(WaterDataset, self).__init__()
        self.split = split
        self.image_size = image_size  # 保存目标尺寸

        self.split_dir = os.path.join(dataset_dir, split)
        suffix = "_" + split

        self.raw_dir = os.path.join(self.split_dir, "input" + suffix)
        self.wb_dir = os.path.join(self.split_dir, "input_wb" + suffix)
        self.ce_dir = os.path.join(self.split_dir, "input_ce" + suffix)
        self.gc_dir = os.path.join(self.split_dir, "input_gc" + suffix)
        self.gt_dir = os.path.join(self.split_dir, "gt" + suffix)

        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f"找不到文件夹: {self.raw_dir}")

        self.filenames = sorted(os.listdir(self.raw_dir))
        self.filenames = [x for x in self.filenames if x.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # 定义读取并缩放的函数 (这部分不需要动)
        def read_and_resize(dir_path, filename):
            path = os.path.join(dir_path, filename)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {path}")
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取并统一尺寸 (这部分不需要动)
        raw = read_and_resize(self.raw_dir, fname)
        wb = read_and_resize(self.wb_dir, fname)
        ce = read_and_resize(self.ce_dir, fname)
        gc = read_and_resize(self.gc_dir, fname)
        gt = read_and_resize(self.gt_dir, fname)

        # --- 修改开始：全部改为 [0, 1] 归一化 ---

        # 原代码: raw = (raw.astype(np.float32) / 127.5) - 1.0  <-- 删掉，这会产生负数
        # 修改为:
        raw = raw.astype(np.float32) / 255.0
        wb = wb.astype(np.float32) / 255.0
        ce = ce.astype(np.float32) / 255.0
        gc = gc.astype(np.float32) / 255.0

        # GT (Ground Truth)
        # 原代码已经是 / 255.0 了，这非常棒，保持不变！
        gt = gt.astype(np.float32) / 255.0

        # --- 修改结束 ---

        # 转 Tensor: (H, W, C) -> (C, H, W) (这部分不需要动)
        raw = torch.from_numpy(raw.transpose(2, 0, 1))
        wb = torch.from_numpy(wb.transpose(2, 0, 1))
        ce = torch.from_numpy(ce.transpose(2, 0, 1))
        gc = torch.from_numpy(gc.transpose(2, 0, 1))
        gt = torch.from_numpy(gt.transpose(2, 0, 1))

        return raw, wb, ce, gc, gt, fname