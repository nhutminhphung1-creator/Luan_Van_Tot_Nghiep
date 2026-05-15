"""
Phân tích các mẫu bị predict sai để hiểu rõ điểm yếu của model.
Kết quả dùng trực tiếp cho phần "Hạn chế" trong báo cáo.
"""
import torch
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns

from dataset import TripletDataset
from model   import JointAnomalyDetector
from loss    import AnomalyLoss
from train   import get_class_weights


def analyze_errors(model, dataset, device, threshold=0.40):
    """Phân loại từng mẫu: TP, TN, FP, FN."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in loader:
            gps    = batch["gps"].to(device)
            i512   = batch["imgs_512"].to(device)
            i256   = batch["imgs_256"].to(device)
            labels = batch["label"]
            ids    = batch["id"]

            out   = model(gps, i512, i256)
            probs = torch.softmax(out["logits"], dim=1)[:, 1].cpu()
            preds = (probs >= threshold).long()

            for j in range(len(labels)):
                gt   = labels[j].item()
                pred = preds[j].item()
                prob = probs[j].item()
                sid  = ids[j] if isinstance(ids[j], str) else ids[j].item()

                if gt == 1 and pred == 1:   cat = "TP"
                elif gt == 0 and pred == 0: cat = "TN"
                elif gt == 0 and pred == 1: cat = "FP"
                else:                       cat = "FN"  # gt=1, pred=0

                results.append({
                    "id": sid, "gt": gt, "pred": pred,
                    "prob": prob, "category": cat,
                    # Lấy GPS meta từ dataset
                    "deviation": gps[j % len(gps), 12].item(),
                    "avg_speed": gps[j % len(gps), 13].item(),
                    "sudden_stop": gps[j % len(gps), 14].item(),
                })
    return results


def plot_error_analysis(results, save_dir="explain_output"):
    cats = {"TP": [], "TN": [], "FP": [], "FN": []}
    for r in results:
        cats[r["category"]].append(r["prob"])

    tp = len(cats["TP"]); tn = len(cats["TN"])
    fp = len(cats["FP"]); fn = len(cats["FN"])
    total_pos = tp + fn
    total_neg = tn + fp

    print("╔══════════════════════════════════════╗")
    print("║         ERROR ANALYSIS               ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  TP (Abnormal đúng) : {tp:>3} / {total_pos}          ║")
    print(f"║  FN (Abnormal sai)  : {fn:>3} / {total_pos}          ║")
    print(f"║  TN (Normal đúng)   : {tn:>3} / {total_neg}          ║")
    print(f"║  FP (Normal sai)    : {fp:>3} / {total_neg}          ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Recall Abnormal    : {tp/(tp+fn):.3f}               ║")
    print(f"║  Precision Abnormal : {tp/(tp+fp) if tp+fp>0 else 0:.3f}               ║")
    print("╚══════════════════════════════════════╝")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d0d1a')

    # ── Plot 1: Confusion Matrix ──
    ax = axes[0]
    ax.set_facecolor('#1e1e2e')
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Normal', 'Pred Abnormal'], color='white', fontsize=10)
    ax.set_yticklabels(['GT Normal', 'GT Abnormal'], color='white', fontsize=10)
    ax.set_title('Confusion Matrix', color='white', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({100*cm[i,j]/cm[i].sum():.0f}%)',
                    ha='center', va='center', color='white', fontsize=11,
                    fontweight='bold')
    plt.colorbar(im, ax=ax)

    # ── Plot 2: Prob distribution theo category ──
    ax2 = axes[1]
    ax2.set_facecolor('#1e1e2e')
    colors = {"TP": "#4CAF50", "TN": "#2196F3", "FP": "#FF9800", "FN": "#F44336"}
    for cat, color in colors.items():
        if cats[cat]:
            ax2.hist(cats[cat], bins=15, alpha=0.7, color=color,
                     label=f'{cat} (n={len(cats[cat])})', density=True)
    ax2.axvline(0.40, color='#FFD700', linestyle='--', lw=2, label='Threshold=0.40')
    ax2.set_xlabel('P(Abnormal)', color='white', fontsize=11)
    ax2.set_ylabel('Density', color='white', fontsize=11)
    ax2.set_title('Prob Distribution\nby Category', color='white', fontsize=12)
    ax2.legend(facecolor='#2d2d3e', labelcolor='white', fontsize=8)
    ax2.tick_params(colors='white')
    for s in ax2.spines.values(): s.set_edgecolor('#444')

    # ── Plot 3: FN analysis — deviation_ratio của mẫu bị bỏ sót ──
    ax3 = axes[2]
    ax3.set_facecolor('#1e1e2e')
    fn_devs = [r["deviation"] for r in results if r["category"] == "FN"]
    tp_devs = [r["deviation"] for r in results if r["category"] == "TP"]
    ax3.hist(tp_devs, bins=15, alpha=0.7, color='#4CAF50',
             label=f'TP Abnormal (detected)\nn={len(tp_devs)}', density=True)
    ax3.hist(fn_devs, bins=15, alpha=0.7, color='#F44336',
             label=f'FN Abnormal (missed)\nn={len(fn_devs)}', density=True)
    ax3.axvline(1.5, color='#FFD700', linestyle='--', lw=2,
                label='Label threshold=1.5')
    ax3.set_xlabel('Deviation Ratio', color='white', fontsize=11)
    ax3.set_ylabel('Density', color='white', fontsize=11)
    ax3.set_title('Missed Abnormal:\nDeviation Ratio Analysis', color='white', fontsize=12)
    ax3.legend(facecolor='#2d2d3e', labelcolor='white', fontsize=8)
    ax3.tick_params(colors='white')
    for s in ax3.spines.values(): s.set_edgecolor('#444')

    plt.tight_layout(pad=2)
    path = f"{save_dir}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {path}")


if __name__ == "__main__":
    from pathlib import Path
    Path("explain_output").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = JointAnomalyDetector(embed_dim=128).to(device)
    ckpt   = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = TripletDataset("dataset/samples.json", split="test")
    results = analyze_errors(model, test_ds, device, threshold=0.40)
    plot_error_analysis(results, save_dir="explain_output")

    # In chi tiết các FN (mẫu Abnormal bị bỏ sót)
    fn_samples = [r for r in results if r["category"] == "FN"]
    print(f"\n── Chi tiết {len(fn_samples)} mẫu FN (bị bỏ sót) ──")
    print(f"{'ID':>8} | {'Prob':>6} | {'DevRatio':>9} | {'AvgSpd':>7} | {'Stop':>5}")
    print("-" * 50)
    for r in sorted(fn_samples, key=lambda x: x["prob"], reverse=True):
        print(f"{str(r['id']):>8} | {r['prob']:>6.3f} | "
              f"{r['deviation']:>9.3f} | {r['avg_speed']:>7.2f} | "
              f"{bool(r['sudden_stop']):>5}")