# cifar10_cnn_cuda_optimized.py
import argparse, os, math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms

# -------------------------
# Utilities
# -------------------------
def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep False for speed with benchmark
    # We aim for speed; use seeds but allow non-deterministic ops for performance.

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# -------------------------
# Model
# -------------------------
class SmallNet(nn.Module):
    """Compact, strong CNN for CIFAR-10: BN + ReLU + GAP + Dropout head"""
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True)
            )
        self.features = nn.Sequential(
            block(3, 64), block(64, 64), nn.MaxPool2d(2),          # 32->16
            block(64, 128), block(128, 128), nn.MaxPool2d(2),      # 16->8
            block(128, 256), block(256, 256)                        # stay 8x8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# -------------------------
# Training / Eval
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, use_amp=True):
    model.train()
    running_loss = 0.0
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, num_classes=10):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    with autocast(enabled=use_amp):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confmat[t, p] += 1
    avg_loss = total_loss / total
    acc = correct / total
    per_class_acc = confmat.diag() / confmat.sum(1).clamp_min(1)
    return avg_loss, acc, confmat, per_class_acc

def save_checkpoint(path, model, optimizer, epoch, best_val_acc, args, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "args": vars(args)
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN (CUDA-optimized)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_split", type=float, default=0.1, help="fraction of train as validation")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_patience", type=int, default=15)
    parser.add_argument("--no_amp", action="store_true", help="disable AMP mixed precision")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--name", type=str, default="smallnet_cifar10")
    args = parser.parse_args()

    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{now()}] Using device: {device}")

    # Speed features
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        # Allow TF32 on Tensor Cores (Ampere+); substantial speed with minimal accuracy hit
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # CIFAR-10 stats + augmentation
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets
    full_train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_tf)
    testset = datasets.CIFAR10(args.data_root, train=False, download=True, transform=test_tf)

    # Split train/val
    val_size = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    trainset, valset = torch.utils.data.random_split(full_train, [train_size, val_size],
                                                     generator=torch.Generator().manual_seed(args.seed))

    # DataLoaders (fast input pipeline)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=max(256, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    # Model, loss, optimizer, scheduler, scaler
    model = SmallNet(num_classes=10, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(trainloader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0
    )
    scaler = GradScaler(enabled=(not args.no_amp) and device.type == "cuda")

    # Training loop with early stopping + checkpointing
    best_val_acc = 0.0
    best_epoch = 0
    patience = args.early_patience
    ckpt_path = os.path.join(args.out_dir, f"{args.name}_best.pt")

    print(f"[{now()}] Start training for {args.epochs} epochs; steps/epoch={steps_per_epoch}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, scheduler, device, scaler, use_amp=not args.no_amp
        )
        val_loss, val_acc, _, _ = evaluate(
            model, valloader, criterion, device, use_amp=not args.no_amp
        )
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        print(f"[{now()}] Epoch {epoch:03d}/{args.epochs} | "
              f"lr={lr:.5f} | train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc*100:.2f}%")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_acc, args, scheduler)
        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"[{now()}] Early stopping at epoch {epoch} (best @ {best_epoch}, val_acc={best_val_acc*100:.2f}%)")
            break

    # Load best and evaluate on test set
    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        print(f"[{now()}] Loaded best checkpoint from epoch {payload['epoch']} (val_acc={payload['best_val_acc']*100:.2f}%)")

    test_loss, test_acc, confmat, per_class_acc = evaluate(
        model, testloader, criterion, device, use_amp=not args.no_amp
    )
    print(f"[{now()}] Test: loss={test_loss:.4f} acc={test_acc*100:.2f}%")
    print("Per-class accuracy (%):")
    for i, a in enumerate(per_class_acc.tolist()):
        print(f"  Class {i}: {a*100:.2f}%")

    # (Optional) save confusion matrix
    cm_path = os.path.join(args.out_dir, f"{args.name}_confmat.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(confmat.cpu(), cm_path)
    print(f"[{now()}] Saved confusion matrix tensor to {cm_path}")

if __name__ == "__main__":
    main()
