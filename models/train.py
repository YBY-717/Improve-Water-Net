import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
from torch.utils.data import DataLoader
from torchvision import models
from model import ImprovedWaterNet
from dataset import WaterDataset


# --- 1. 定义 Perceptual Loss ---
class PerceptualLoss(torch.nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_sub = torch.nn.Sequential(*list(vgg.children())[:35])
        self.vgg_sub.eval()
        for param in self.vgg_sub.parameters():
            param.requires_grad = False
        self.vgg_sub = self.vgg_sub.to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def normalize_vgg(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, gt):
        pred_norm = self.normalize_vgg(pred)
        gt_norm = self.normalize_vgg(gt)
        pred_feat = self.vgg_sub(pred_norm)
        gt_feat = self.vgg_sub(gt_norm)
        return torch.mean((pred_feat - gt_feat) ** 2)


# --- 2. 定义验证函数 (已移除 PSNR) ---
def validate(model, val_loader, criterion_l1, criterion_vgg, device, texture_weight):
    """
    运行验证集评估，仅计算 Loss
    """
    model.eval()  # 切换到评估模式
    total_val_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():  # 不计算梯度
        for raw, wb, ce, gc, gt, _ in val_loader:
            raw = raw.to(device)
            wb = wb.to(device)
            ce = ce.to(device)
            gc = gc.to(device)
            gt = gt.to(device)

            # Forward
            output = model(raw, wb, ce, gc)

            # --- 修改为 ---
            loss_pixel = criterion_l1(output, gt)  # gt 本身就是 0-1
            out_01 = torch.clamp(output, 0, 1)
            out_01 = torch.clamp(out_01, 0, 1)
            loss_texture = criterion_vgg(out_01, gt)

            loss = loss_pixel + texture_weight * loss_texture
            total_val_loss += loss.item()

    avg_loss = total_val_loss / num_batches
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='ImprovedWaterNet Training')
    # 训练参数
    parser.add_argument('-b', '--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    # 模型参数
    parser.add_argument('--texture_weight', type=float, default=0.05, help='weight for texture loss')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum lr')
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data_UIEB', help='dataset directory')
    parser.add_argument('--image_size', type=int, default=256, help='input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save directory')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--resume_best', action='store_true', help='resume from best models')
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser.parse_args()


# --- Training Loop ---
def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 准备数据 (Train Set)
    print("Loading Training Set...")
    train_dataset = WaterDataset(args.data_dir, split='train', image_size=(args.image_size, args.image_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    # 2. 准备数据 (Validation Set)
    print("Loading Validation Set...")
    val_dataset = WaterDataset(args.data_dir, split='val', image_size=(args.image_size, args.image_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 3. 模型与优化器
    model = ImprovedWaterNet().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    criterion_l1 = torch.nn.L1Loss()
    criterion_vgg = PerceptualLoss(device)

    # 4. 初始化最佳记录
    best_val_loss = float('inf')
    start_epoch = 0

    if args.resume:
        checkpoint_path = args.resume
    elif args.resume_best:
        checkpoint_path = os.path.join(args.save_dir, 'ImprovedWaterNet_best.pth')
    else:
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, previous best val loss: {best_val_loss:.4f}")

        # =======================================================
        # ### 新增代码：强制更新学习率为命令行参数的值 ###
        # =======================================================
        print(f"Overriding learning rate with args.lr: {args.lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        # 如果你希望 Scheduler 也基于这个新 LR 重新开始计算（防止被 Scheduler 又改回去）
        # 你可能还需要更新 scheduler 的基础学习率，或者直接不加载 scheduler 的 state_dict
        # 简单粗暴的方法是同步更新 scheduler 的 base_lrs：
        for i in range(len(scheduler.base_lrs)):
            scheduler.base_lrs[i] = args.lr

        # ★★★ 关键修改：告诉 Scheduler 我们目前在哪个 epoch，确保余弦退火从正确位置继续 ★★★
        scheduler.last_epoch = start_epoch - 1
        # =======================================================

    print("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # --- 训练阶段 ---
        model.train()
        epoch_train_loss = 0

        for i, (raw, wb, ce, gc, gt, _) in enumerate(train_loader):
            raw, wb, ce, gc, gt = raw.to(device), wb.to(device), ce.to(device), gc.to(device), gt.to(device)

            optimizer.zero_grad()
            output = model(raw, wb, ce, gc)

            # --- 修改为 (保持 [0, 1] 范围) ---
            # 不需要对 gt 做任何处理，只要保证 gt 读进来是 0-1 即可（PyTorch ToTensor默认就是0-1）
            loss_pixel = criterion_l1(output, gt)

            # --- 修改为 ---
            # 因为用了 ReLU，output可能会稍微超过1，所以只做 clamp 即可，不需要反归一化
            out_01 = torch.clamp(output, 0, 1)
            loss_texture = criterion_vgg(out_01, gt)

            loss = loss_pixel + args.texture_weight * loss_texture
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"[Train] Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        # --- 验证阶段 (已移除 PSNR) ---
        print(f"Validating epoch {epoch + 1}...")
        avg_val_loss = validate(model, val_loader, criterion_l1, criterion_vgg, device, args.texture_weight)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{args.epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # --- 保存逻辑 ---
        # 1. 保存 Recent
        recent_path = os.path.join(args.save_dir, 'ImprovedWaterNet_recent.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
            'args': vars(args)
        }, recent_path)

        # 2. 保存 Best (根据 Val Loss)
        if avg_val_loss < best_val_loss:
            print(f"Validation Loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving best models...")
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, 'ImprovedWaterNet_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'args': vars(args)
            }, best_path)

    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    args = parse_args()
    train(args)

# 需要先使用 data_UIEB/train 和 data_UIEB/val下的generate脚本生成对应的ce、wb、gc图片
# --data_dir后面填入图片的总目录，后续会根据该总目录去找里边的train、val相应的图片进行学习