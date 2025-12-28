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
    parser = argparse.ArgumentParser(description='ImprovedWaterNet Training & Ablation')

    # --- 消融实验控制开关 (默认为不禁用，即全开启) ---
    parser.add_argument('--no_transformer', action='store_true', help='Disable Transformer Module')
    parser.add_argument('--no_cbam', action='store_true', help='Disable CBAM Module')
    parser.add_argument('--no_aspp', action='store_true', help='Disable ASPP Module (use Conv3x3 instead)')

    # ... (保留原有的其他参数: batch_size, epochs, lr 等) ...
    parser.add_argument('-b', '--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--texture_weight', type=float, default=0.05, help='weight for texture loss')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum lr')
    parser.add_argument('--data_dir', type=str, default='data_UIEB', help='dataset directory')
    parser.add_argument('--image_size', type=int, default=256, help='input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save directory')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--resume_best', action='store_true', help='resume from best models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser.parse_args()


def train(args):
    # ... (Seed 和 Device 设置保持不变) ...
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # --- 1. 生成实验名称 (用于区分保存的文件) ---
    # 逻辑: 如果是 Full 模型，名字叫 "Full"；否则标出缺少的模块
    exp_name_parts = []
    if args.no_transformer: exp_name_parts.append("NoTrans")
    if args.no_cbam: exp_name_parts.append("NoCBAM")
    if args.no_aspp: exp_name_parts.append("NoASPP")

    if len(exp_name_parts) == 0:
        exp_name = "Full_Model"
    else:
        exp_name = "_".join(exp_name_parts)

    print(f"==========================================")
    print(f"Running Experiment: {exp_name}")
    print(f"Transformer: {not args.no_transformer} | CBAM: {not args.no_cbam} | ASPP: {not args.no_aspp}")
    print(f"==========================================")

    os.makedirs(args.save_dir, exist_ok=True)

    # ... (Dataset 和 DataLoader 部分保持不变) ...
    print("Loading Training Set...")
    train_dataset = WaterDataset(args.data_dir, split='train', image_size=(args.image_size, args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_dataset = WaterDataset(args.data_dir, split='val', image_size=(args.image_size, args.image_size))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # --- 2. 实例化模型 (传入参数) ---
    model = ImprovedWaterNet(
        use_transformer=not args.no_transformer,
        use_cbam=not args.no_cbam,
        use_aspp=not args.no_aspp
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    criterion_l1 = torch.nn.L1Loss()
    criterion_vgg = PerceptualLoss(device)

    # --- 3. Resume 逻辑 (更新文件名) ---
    best_val_loss = float('inf')
    start_epoch = 0

    # 自动定义文件名
    best_model_name = f'ImprovedWaterNet_{exp_name}_best.pth'
    recent_model_name = f'ImprovedWaterNet_{exp_name}_recent.pth'

    if args.resume:
        checkpoint_path = args.resume
    elif args.resume_best:
        checkpoint_path = os.path.join(args.save_dir, best_model_name)
    else:
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 这里要加 strict=False，以防你在不同实验间切换加载权重导致 key 不匹配报错
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('loss', float('inf'))
        scheduler.last_epoch = start_epoch - 1
        print(f"Resumed from epoch {start_epoch}, best loss: {best_val_loss:.4f}")

    # ... (Training Loop 保持不变) ...
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_train_loss = 0

        for i, (raw, wb, ce, gc, gt, _) in enumerate(train_loader):
            raw, wb, ce, gc, gt = raw.to(device), wb.to(device), ce.to(device), gc.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(raw, wb, ce, gc)

            loss_pixel = criterion_l1(output, gt)
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

        print(f"Validating epoch {epoch + 1}...")
        avg_val_loss = validate(model, val_loader, criterion_l1, criterion_vgg, device, args.texture_weight)

        print(f"Epoch [{epoch + 1}/{args.epochs}] ({exp_name}) Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # --- 保存逻辑 (使用动态文件名) ---
        recent_path = os.path.join(args.save_dir, recent_model_name)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
            'args': vars(args)
        }, recent_path)

        if avg_val_loss < best_val_loss:
            print(f"Validation Loss improved. Saving to {best_model_name}...")
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.save_dir, best_model_name)
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
