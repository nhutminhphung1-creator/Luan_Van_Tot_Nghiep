import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

from dataset import TripletDataset
from model   import JointAnomalyDetector
from loss    import AnomalyLoss

# ─── Config ───────────────────────────────────────────
CFG = {
    "json_path":   "dataset/samples.json",
    "epochs":      50,
    "batch_size":  8,
    "lr":          1e-4,          # ← giảm từ 2e-4
    "embed_dim":   128,
    "lambda_c":    0.5,           # ← tăng từ 0.3 (contrastive loss mạnh hơn)
    "patience":    15,            # ← tăng từ 10
    "save_dir":    "checkpoints",
    "device":      "cuda" if torch.cuda.is_available() else "cpu",

    # Threshold tuning: thay vì dùng 0.5 mặc định
    # model sẽ predict Abnormal nếu prob > threshold này
    "abnormal_threshold": 0.40,   # ← thấp hơn 0.5 → bắt được nhiều Abnormal hơn
}

def get_class_weights(dataset):
    """Tính class weight để xử lý imbalance (2515 Normal vs 342 Abnormal)."""
    labels = [s["label"] for s in dataset.samples]
    n0 = labels.count(0)
    n1 = labels.count(1)
    total = n0 + n1
    # Weight nghịch đảo với tần suất
    w = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float)
    return w

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = ce_loss = ct_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        gps      = batch["gps"].to(device)
        imgs_512 = batch["imgs_512"].to(device)
        imgs_256 = batch["imgs_256"].to(device)
        labels   = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(gps, imgs_512, imgs_256)
        losses  = criterion(outputs, labels)

        losses["total"].backward()
        # Gradient clipping — quan trọng với Transformer
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total"].item()
        ce_loss    += losses["ce"].item()
        ct_loss    += losses["contrast"].item()

        preds = outputs["logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {"loss": total_loss/n, "ce": ce_loss/n, "contrast": ct_loss/n, "f1": f1}

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        gps      = batch["gps"].to(device)
        imgs_512 = batch["imgs_512"].to(device)
        imgs_256 = batch["imgs_256"].to(device)
        labels   = batch["label"].to(device)

        outputs = model(gps, imgs_512, imgs_256)
        losses  = criterion(outputs, labels)
        total_loss += losses["total"].item()

        preds = outputs["logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {"loss": total_loss/len(loader), "f1": f1,
            "report": classification_report(all_labels, all_preds,
                       target_names=["Normal","Abnormal"], zero_division=0)}

def eval_with_threshold(model, loader, criterion, device, threshold=0.35):
    """
    Thay vì argmax, dùng threshold trên xác suất Abnormal.
    Giúp tăng Recall Abnormal (quan trọng hơn Precision trong bài toán an toàn).
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            gps      = batch["gps"].to(device)
            imgs_512 = batch["imgs_512"].to(device)
            imgs_256 = batch["imgs_256"].to(device)
            labels   = batch["label"].to(device)

            outputs = model(gps, imgs_512, imgs_256)
            losses  = criterion(outputs, labels)
            total_loss += losses["total"].item()

            # Lấy xác suất class Abnormal (index 1)
            probs = torch.softmax(outputs["logits"], dim=1)[:, 1]
            # Predict Abnormal nếu prob >= threshold
            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=["Normal", "Abnormal"], zero_division=0
    )
    return {
        "loss": total_loss / len(loader),
        "f1": f1,
        "report": report,
        "probs": all_probs,
        "labels": all_labels,
    }

def main():
    device = CFG["device"]
    print(f"Device: {device}")
    Path(CFG["save_dir"]).mkdir(exist_ok=True)

    # Dataset
    train_ds = TripletDataset(CFG["json_path"], split="train")
    val_ds   = TripletDataset(CFG["json_path"], split="val")
    test_ds  = TripletDataset(CFG["json_path"], split="test")

    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=2)

    # Model
    model = JointAnomalyDetector(embed_dim=CFG["embed_dim"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # Loss với class weights (xử lý imbalance)
    cw = get_class_weights(train_ds).to(device)
    print(f"Class weights: Normal={cw[0]:.2f}, Abnormal={cw[1]:.2f}")
    criterion = AnomalyLoss(lambda_contrast=CFG["lambda_c"], class_weights=cw)

    # Optimizer: AdamW với learning rate thấp hơn cho backbone pretrained
    backbone_params = (list(model.visual_encoder.backbone_512.parameters()) +
                       list(model.visual_encoder.backbone_256.parameters()))
    other_params    = [p for p in model.parameters()
                       if not any(p is q for q in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": CFG["lr"] * 0.1},  # fine-tune nhẹ
        {"params": other_params,    "lr": CFG["lr"]},
    ], weight_decay=1e-4)

    # Scheduler: Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )

    # Training loop
    best_val_f1 = 0
    patience_cnt = 0

    for epoch in range(1, CFG["epochs"] + 1):
        tr = train_epoch(model, train_dl, optimizer, criterion, device)
        va = eval_epoch(model, val_dl, criterion, device)
        scheduler.step()

        print(f"Ep {epoch:02d} | "
              f"Train loss={tr['loss']:.4f} f1={tr['f1']:.3f} | "
              f"Val loss={va['loss']:.4f} f1={va['f1']:.3f}")

        # Save best
        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            patience_cnt = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_f1": best_val_f1, "cfg": CFG
            }, f"{CFG['save_dir']}/best_model.pt")
            print(f"  ✅ Saved best model (val F1={best_val_f1:.3f})")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    # ─── BẮT ĐẦU PHẦN SỬA ĐỔI 2: TEST VỚI THRESHOLD ──────────────────────────
    print("\n── TEST RESULTS (threshold=0.35) ──")
    ckpt = torch.load(f"{CFG['save_dir']}/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_dl = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=2)
    
    # Sử dụng hàm eval_with_threshold thay vì eval_epoch
    test_res = eval_with_threshold(model, test_dl, criterion, device,
                                    threshold=CFG["abnormal_threshold"])
    print(test_res["report"])
    # ─── KẾT THÚC PHẦN SỬA ĐỔI 2 ─────────────────────────────────────────────

if __name__ == "__main__":
    main()