import os
import yaml
import datetime
import argparse
import csv
import random
from collections import Counter
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
from scipy import stats
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.MSHF import MSHF
try:
    from models.MSHF_ViT import MSHF_ViT
except ImportError:
    MSHF_ViT = None
from dataloader.load_data import split_dataset, k_fold_split_dataset, MyDataset
from utils.losses import FocalLoss


METRIC_NAMES = ["Acc", "F1", "Rec", "Pre", "AUC"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_clinical_path(config, config_path):
    clinical_path = config["clinical_dir"]
    if os.path.isabs(clinical_path):
        return clinical_path

    config_dir_path = os.path.join(os.path.dirname(config_path), clinical_path)
    if os.path.exists(config_dir_path):
        return config_dir_path

    sibling_path = os.path.join(os.path.dirname(config_path), "clinical.json")
    if os.path.exists(sibling_path):
        return sibling_path

    return clinical_path


def build_model(args, device):
    if args.model == "MSHF_ViT":
        if MSHF_ViT is None:
            raise ImportError("models/MSHF_ViT.py was not found, so --model MSHF_ViT cannot be used.")
        model = MSHF_ViT(num_classes=args.num_classes, backbone=args.backbone)
    else:
        model = MSHF(num_classes=args.num_classes, backbone=args.backbone)

    return model.to(device)


def freeze_backbone_layers(model):
    for backbone in [model.mg_backbone, model.us_backbone]:
        for name, param in backbone.named_parameters():
            first_part = name.split(".")[0]
            if "7" in first_part or "denseblock4" in name or "norm5" in name or "Mixed_7" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


def build_optimizer(args, model):
    head_params = [p for n, p in model.named_parameters() if "mg_backbone" not in n and "us_backbone" not in n]
    mg_params = [p for p in model.mg_backbone.parameters() if p.requires_grad]
    us_params = [p for p in model.us_backbone.parameters() if p.requires_grad]

    return optim.AdamW(
        [
            {"params": mg_params, "lr": 1e-6},
            {"params": us_params, "lr": 1e-6},
            {"params": head_params, "lr": args.lr},
        ],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )


def build_criterion(args, train_info, device):
    train_labels = [info["label"] for info in train_info]
    counts = Counter(train_labels)
    total = sum(counts.values())
    alpha_vals = [total / (args.num_classes * counts.get(i, 1)) for i in range(args.num_classes)]
    alpha_sum = sum(alpha_vals)
    alpha = torch.tensor([v / alpha_sum for v in alpha_vals], dtype=torch.float32).to(device)
    return FocalLoss(alpha=alpha, gamma=2)


def build_train_sampler(train_info):
    train_labels = [info["label"] for info in train_info]
    counts = Counter(train_labels)
    class_sample_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = [class_sample_weights[label] for label in train_labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_metrics(labels, preds, probs, num_classes):
    metrics = {
        "Acc": accuracy_score(labels, preds),
        "F1": f1_score(labels, preds, average="macro", zero_division=0),
        "Rec": recall_score(labels, preds, average="macro", zero_division=0),
        "Pre": precision_score(labels, preds, average="macro", zero_division=0),
        "AUC": 0.0,
    }

    try:
        probs = np.asarray(probs)
        if num_classes == 2:
            metrics["AUC"] = roc_auc_score(labels, probs[:, 1])
        else:
            metrics["AUC"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["AUC"] = 0.0

    return metrics


def update_writer(writer, prefix, loss, metrics, epoch):
    writer.add_scalar(f"Loss/{prefix}", loss, epoch)
    writer.add_scalar(f"Accuracy/{prefix}", metrics["Acc"], epoch)
    writer.add_scalar(f"F1/{prefix}", metrics["F1"], epoch)
    writer.add_scalar(f"Recall/{prefix}", metrics["Rec"], epoch)
    writer.add_scalar(f"Precision/{prefix}", metrics["Pre"], epoch)
    writer.add_scalar(f"AUC/{prefix}", metrics["AUC"], epoch)


def unpack_batch(batch, device):
    (img_cc, mask_cc), (img_mlo, mask_mlo), (img_us, mask_us), labels, clinical = batch
    return (
        img_cc.to(device),
        mask_cc.to(device),
        img_mlo.to(device),
        mask_mlo.to(device),
        img_us.to(device),
        mask_us.to(device),
        labels.to(device),
        clinical.to(device),
    )


def format_metrics(metrics):
    return ", ".join([f"{name}: {metrics[name]:.4f}" for name in METRIC_NAMES])


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0

    sem = stats.sem(values)
    half_width = float(sem * stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean, half_width

def train_one_epoch(model, loader, dataset_size, criterion, optimizer, scaler, device, num_classes, epoch, epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
    for batch in loop:
        img_cc, mask_cc, img_mlo, mask_mlo, img_us, mask_us, labels, clinical = unpack_batch(batch, device)

        optimizer.zero_grad()
        with autocast(enabled=device.type == "cuda"):
            outputs = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
            if isinstance(outputs, tuple):
                out_main, out_mg, out_us = outputs
                loss = criterion(out_main, labels) + 0.3 * criterion(out_mg, labels) + 0.3 * criterion(out_us, labels)
                logits = out_main
            else:
                loss = criterion(outputs, labels)
                logits = outputs

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / dataset_size
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    return epoch_loss, metrics


def validate(model, loader, dataset_size, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            img_cc, mask_cc, img_mlo, mask_mlo, img_us, mask_us, labels, clinical = unpack_batch(batch, device)

            outputs = model(img_mlo, img_cc, img_us, clinical, mask_mlo, mask_cc, mask_us)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    val_loss = running_loss / dataset_size
    metrics = compute_metrics(all_labels, all_preds, all_probs, num_classes)
    return val_loss, metrics


def summarize_cross_validation(fold_metrics, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)
    summary = {}
    summary_csv = os.path.join(summary_dir, "cv_summary_metrics.csv")
    summary_txt = os.path.join(summary_dir, "cv_summary_metrics.txt")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "95CI", "Mean±95CI"])
        for name in METRIC_NAMES:
            values = [metrics[name] for metrics in fold_metrics]
            mean, ci = confidence_interval(values)
            summary[name] = (mean, ci)
            writer.writerow([name, mean, ci, f"{mean:.4f}±{ci:.4f}"])

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Cross-validation summary (mean±95%CI)\n")
        for name in METRIC_NAMES:
            mean, ci = summary[name]
            f.write(f"{name}: {mean:.4f}±{ci:.4f}\n")

    print("\nCross-validation summary (mean±95%CI):")
    for name in METRIC_NAMES:
        mean, ci = summary[name]
        print(f"{name}: {mean:.4f}±{ci:.4f}")

    return summary


def train_fold(args, fold_idx, train_info, val_info, experiment_name, device):
    print(f"\n===== Fold {fold_idx} / {args.cv_folds} =====")
    print(f"Train samples: {len(train_info)}, Val samples: {len(val_info)}")
    print(f"Train label distribution: {dict(Counter([info['label'] for info in train_info]))}")
    print(f"Val label distribution: {dict(Counter([info['label'] for info in val_info]))}")

    train_dataset = MyDataset(train_info, args.config_path, use_seg=True, is_train=True)
    val_dataset = MyDataset(val_info, args.config_path, use_seg=True, is_train=False)

    sampler = build_train_sampler(train_info)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, device)
    freeze_backbone_layers(model)
    optimizer = build_optimizer(args, model)
    criterion = build_criterion(args, train_info, device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-7
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    fold_name = f"{experiment_name}_fold{fold_idx}"
    save_dir = os.path.join(args.output_dir, "checkpoints", fold_name)
    log_dir = os.path.join(args.output_dir, "runs", fold_name)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    csv_path = os.path.join(save_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "Epoch",
                "Train_Loss",
                "Train_Acc",
                "Train_F1",
                "Train_Rec",
                "Train_Pre",
                "Train_AUC",
                "Val_Loss",
                "Val_Acc",
                "Val_F1",
                "Val_Rec",
                "Val_Pre",
                "Val_AUC",
            ]
        )

    best_val_auc = -1.0
    best_metrics = None
    best_epoch = 0

    for epoch in range(args.num_epochs):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, len(train_dataset), criterion, optimizer, scaler, device, args.num_classes, epoch, args.num_epochs
        )
        val_loss, val_metrics = validate(model, val_loader, len(val_dataset), criterion, device, args.num_classes)

        print(f"Epoch {epoch + 1} Results:")
        print(f"Train - Loss: {train_loss:.4f}, {format_metrics(train_metrics)}")
        print(f"Val   - Loss: {val_loss:.4f}, {format_metrics(val_metrics)}")

        update_writer(writer, "train", train_loss, train_metrics, epoch)
        update_writer(writer, "val", val_loss, val_metrics, epoch)

        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_metrics["Acc"],
                    train_metrics["F1"],
                    train_metrics["Rec"],
                    train_metrics["Pre"],
                    train_metrics["AUC"],
                    val_loss,
                    val_metrics["Acc"],
                    val_metrics["F1"],
                    val_metrics["Rec"],
                    val_metrics["Pre"],
                    val_metrics["AUC"],
                ]
            )

        scheduler.step(val_metrics["AUC"])

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

            best_metrics_path = os.path.join(save_dir, "best_metrics.txt")
            with open(best_metrics_path, "w", encoding="utf-8") as f_best:
                f_best.write(f"Best Model Metrics (Epoch {best_epoch}):\n")
                for name in METRIC_NAMES:
                    f_best.write(f"{name}: {best_metrics[name]:.4f}\n")

            print("Saved Best Model!")

    writer.close()

    best_metrics["Fold"] = fold_idx
    best_metrics["Best_Epoch"] = best_epoch
    print(f"Fold {fold_idx} best epoch: {best_epoch}, {format_metrics(best_metrics)}")
    return best_metrics


def train(args):
    set_seed(args.seed)
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    clinical_path = resolve_clinical_path(config, args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {clinical_path}")

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{start_time}_{args.model}_{args.backbone}"

    if args.cv_folds > 1:
        folds = k_fold_split_dataset(clinical_path, n_splits=args.cv_folds, seed=args.seed)
    else:
        train_info, val_info = split_dataset(clinical_path, ratio=args.train_ratio, seed=args.seed)
        folds = [(train_info, val_info)]
        args.cv_folds = 1

    fold_metrics = []
    for fold_idx, (train_info, val_info) in enumerate(folds, start=1):
        fold_seed = args.seed + fold_idx - 1
        set_seed(fold_seed)
        metrics = train_fold(args, fold_idx, train_info, val_info, experiment_name, device)
        fold_metrics.append(metrics)

    fold_csv = os.path.join(args.output_dir, "checkpoints", experiment_name, "fold_best_metrics.csv")
    os.makedirs(os.path.dirname(fold_csv), exist_ok=True)
    with open(fold_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Best_Epoch", *METRIC_NAMES])
        for metrics in fold_metrics:
            writer.writerow([metrics["Fold"], metrics["Best_Epoch"], *[metrics[name] for name in METRIC_NAMES]])

    if args.cv_folds > 1:
        summarize_cross_validation(fold_metrics, os.path.dirname(fold_csv))

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model', type=str, default='MSHF',
                        choices=['MSHF', 'MSHF_ViT'],
                        help='Model variant to train (MSHF: original, MSHF_ViT: ViT-based fusion)')
    parser.add_argument('--backbone', type=str, default='ResNet50', 
                        choices=['ResNet50', 'DenseNet121', 'InceptionV3'],
                        help='Backbone model for feature extraction')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of stratified cross-validation folds')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio when --cv_folds is 1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--output_dir', type=str, default='.', help='Root directory for checkpoints and TensorBoard runs')
    args = parser.parse_args()
    
    train(args)
