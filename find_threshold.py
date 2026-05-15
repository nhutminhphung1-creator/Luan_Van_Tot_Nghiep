import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve

from dataset import TripletDataset
from model   import JointAnomalyDetector
from loss    import AnomalyLoss

def find_best_threshold(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            gps      = batch["gps"].to(device)
            imgs_512 = batch["imgs_512"].to(device)
            imgs_256 = batch["imgs_256"].to(device)
            labels   = batch["label"]

            outputs = model(gps, imgs_512, imgs_256)
            probs   = torch.softmax(outputs["logits"], dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    # Thử các threshold từ 0.1 → 0.9
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_ab = f1_score(labels, preds, pos_label=1, zero_division=0)
        f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
        results.append({"threshold": t, "f1_abnormal": f1_ab, "f1_macro": f1_mac})
        print(f"  threshold={t:.2f} | F1-Abnormal={f1_ab:.3f} | F1-Macro={f1_mac:.3f}")

    # Tìm threshold tốt nhất theo F1 Abnormal
    best = max(results, key=lambda r: r["f1_abnormal"])
    print(f"\n✅ Best threshold: {best['threshold']:.2f} "
          f"→ F1-Abnormal={best['f1_abnormal']:.3f}")

    # Vẽ Precision-Recall curve
    precision, recall, pr_thresh = precision_recall_curve(labels, probs)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.xlabel('Recall (Abnormal)')
    plt.ylabel('Precision (Abnormal)')
    plt.title('Precision-Recall Curve — Abnormal Class')
    plt.grid(True)
    plt.legend()
    plt.savefig("pr_curve.png", dpi=150, bbox_inches='tight')
    print("📊 Saved: pr_curve.png")
    return best["threshold"]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load val set (không dùng test set để tránh data leakage)
    val_ds = TripletDataset("dataset/samples.json", split="val")
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = JointAnomalyDetector(embed_dim=128).to(device)
    ckpt  = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    best_t = find_best_threshold(model, val_dl, device)
    print(f"\n→ Dùng threshold={best_t:.2f} trong train.py CFG['abnormal_threshold']")