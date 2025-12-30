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


# --- 2. 定义验证函数 ---
def validate(model, val_loader, criterion_l1, criterion_vgg, device, texture_weight):
    model.eval()
    total_val_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for raw, wb, ce, gc, gt, _ in val_loader:
            raw, wb, ce, gc, gt = raw.to(device), wb.to(device), ce.to(device), gc.to(device), gt.to(device)
            
            output = model(raw, wb, ce, gc)
            
            loss_pixel = criterion_l1(output, gt)
            out_01 = torch.clamp(output, 0, 1)
            loss_texture = criterion_vgg(out_01, gt)
            
            loss = loss_pixel + texture_weight * loss_texture
            total_val_loss += loss.item()

    avg_loss = total_val_loss / num_batches
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='ImprovedWaterNet Training & Ablation')

    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--data_dir', type=str, default='data_UIEB', help='dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='save directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--texture_weight', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_best', action='store_true')

    return parser.parse_args()


def train(args):
    # [关键修复] 禁用 cuDNN benchmark 以防止 CUDNN_STATUS_NOT_SUPPORTED 错误
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(">> Note: cuDNN benchmark disabled to ensure stability.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # --- 1. 确定模型配置与实验名称 ---
    use_transformer = True
    use_cbam = True
    use_aspp = True
    exp_name = "Full_Model"

    if args.no_perception:
        use_transformer = False
        use_cbam = False
        exp_name = "NoPerception" # 对应论文中的“去除感知模块”
    
    if args.no_aspp:
        use_aspp = False
        exp_name = "NoASPP"

    print(f"==========================================")
    print(f"Running Experiment: {exp_name}")
    print(f"Config -> Trans: {use_transformer} | CBAM: {use_cbam} | ASPP: {use_aspp}")
    print(f"==========================================")
    
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading Dataset...")
    train_dataset = WaterDataset(args.data_dir, split='train', image_size=(args.image_size, args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_dataset = WaterDataset(args.data_dir, split='val', image_size=(args.image_size, args.image_size))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # --- 2. 实例化模型 ---
    model = ImprovedWaterNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    criterion_l1 = torch.nn.L1Loss()
    criterion_vgg = PerceptualLoss(device)

    # --- 3. Resume 逻辑 ---
    best_val_loss = float('inf')
    start_epoch = 0
    
    # 动态定义文件名
    best_model_name = f'{exp_name}_best.pth'
    recent_model_name = f'{exp_name}_recent.pth'

    if args.resume:
        checkpoint_path = args.resume
    elif args.resume_best:
        checkpoint_path = os.path.join(args.save_dir, best_model_name)
    else:
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('loss', float('inf'))
        scheduler.last_epoch = start_epoch - 1
        print(f"Resumed from epoch {start_epoch}, best loss: {best_val_loss:.4f}")

    # --- 4. 训练循环 ---
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
                print(f"[Train] Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_train_loss / len(train_loader)

        print(f"Validating epoch {epoch + 1}...")
        avg_val_loss = validate(model, val_loader, criterion_l1, criterion_vgg, device, args.texture_weight)

        print(f"Epoch [{epoch + 1}/{args.epochs}] ({exp_name}) Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # 保存 Recent
        recent_path = os.path.join(args.save_dir, recent_model_name)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_val_loss,
            'args': vars(args)
        }, recent_path)

        # 保存 Best
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
