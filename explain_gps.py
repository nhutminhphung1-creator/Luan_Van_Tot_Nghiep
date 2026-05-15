"""
Mahalanobis Distance Score — đo độ lệch GPS so với hành trình bình thường.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, roc_curve

from dataset import TripletDataset
from model   import JointAnomalyDetector


def extract_embeddings(model, loader, device):
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["gps"].to(device),
                batch["imgs_512"].to(device),
                batch["imgs_256"].to(device)
            )
            all_z.extend(out["z_traj"].cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_z), np.array(all_labels)


def fit_normal_distribution(z, labels):
    z_normal = z[labels == 0]
    mu       = z_normal.mean(axis=0)
    cov      = np.cov(z_normal.T) + np.eye(z_normal.shape[1]) * 1e-4
    cov_inv  = np.linalg.inv(cov)
    return mu, cov_inv


def compute_scores(z, mu, cov_inv):
    return np.array([mahalanobis(zi, mu, cov_inv) for zi in z])


def plot_results(scores, labels, save_dir="."):
    normal_s   = scores[labels == 0]
    abnormal_s = scores[labels == 1]
    threshold  = np.percentile(normal_s, 95)

    # ── AUC-ROC ──
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d0d1a')

    # Plot 1: Histogram
    ax = axes[0]
    ax.set_facecolor('#1e1e2e')
    ax.hist(normal_s,   bins=40, alpha=0.75, color='#4CAF50',
            label=f'Normal (n={len(normal_s)})',   density=True)
    ax.hist(abnormal_s, bins=40, alpha=0.75, color='#F44336',
            label=f'Abnormal (n={len(abnormal_s)})', density=True)
    ax.axvline(threshold, color='#FFD700', linestyle='--', lw=2,
               label=f'Threshold={threshold:.1f} (p95)')
    ax.set_xlabel('Mahalanobis Distance', color='white', fontsize=11)
    ax.set_ylabel('Density', color='white', fontsize=11)
    ax.set_title('GPS Anomaly Score\nDistribution', color='white', fontsize=12)
    ax.legend(facecolor='#2d2d3e', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#444')

    # Plot 2: Scatter per sample
    ax2 = axes[1]
    ax2.set_facecolor('#1e1e2e')
    idx_n = np.where(labels == 0)[0]
    idx_a = np.where(labels == 1)[0]
    ax2.scatter(idx_n, scores[idx_n], s=8,  alpha=0.5, color='#4CAF50', label='Normal')
    ax2.scatter(idx_a, scores[idx_a], s=20, alpha=0.9, color='#F44336', label='Abnormal',
                zorder=5)
    ax2.axhline(threshold, color='#FFD700', linestyle='--', lw=1.5,
                label=f'Threshold={threshold:.1f}')
    ax2.set_xlabel('Sample Index', color='white', fontsize=11)
    ax2.set_ylabel('Mahalanobis Distance', color='white', fontsize=11)
    ax2.set_title('Anomaly Score\nper Sample', color='white', fontsize=12)
    ax2.legend(facecolor='#2d2d3e', labelcolor='white', fontsize=9)
    ax2.tick_params(colors='white')
    for s in ax2.spines.values(): s.set_edgecolor('#444')

    # Plot 3: ROC Curve
    ax3 = axes[2]
    ax3.set_facecolor('#1e1e2e')
    ax3.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC = {auc:.3f}')
    ax3.plot([0,1],[0,1], color='#666', linestyle='--', lw=1)
    ax3.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    ax3.set_xlabel('False Positive Rate', color='white', fontsize=11)
    ax3.set_ylabel('True Positive Rate', color='white', fontsize=11)
    ax3.set_title(f'ROC Curve\n(Mahalanobis GPS Score)', color='white', fontsize=12)
    ax3.legend(facecolor='#2d2d3e', labelcolor='white', fontsize=10)
    ax3.tick_params(colors='white')
    for s in ax3.spines.values(): s.set_edgecolor('#444')

    plt.tight_layout(pad=2)
    path = f"{save_dir}/mahalanobis_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {path}")

    # ── In số liệu ──
    tp = np.sum(abnormal_s > threshold)
    fp = np.sum(normal_s   > threshold)
    print(f"\n── Mahalanobis Results (threshold=p95={threshold:.2f}) ──")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Detected Abnormal: {tp}/{len(abnormal_s)} "
          f"({100*tp/len(abnormal_s):.1f}%)")
    print(f"  False Alarm      : {fp}/{len(normal_s)} "
          f"({100*fp/len(normal_s):.1f}%)")
    return threshold, auc


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JointAnomalyDetector(embed_dim=128).to(device)
    ckpt  = torch.load("checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print("✅ Model loaded")

    train_dl = DataLoader(TripletDataset("dataset/samples.json","train"),
                          batch_size=16, shuffle=False, num_workers=0)
    test_dl  = DataLoader(TripletDataset("dataset/samples.json","test"),
                          batch_size=16, shuffle=False, num_workers=0)

    print("Extracting embeddings (train)...")
    z_train, y_train = extract_embeddings(model, train_dl, device)
    mu, cov_inv = fit_normal_distribution(z_train, y_train)
    print(f"  Fitted on {(y_train==0).sum()} Normal samples")

    print("Extracting embeddings (test)...")
    z_test, y_test = extract_embeddings(model, test_dl, device)
    scores = compute_scores(z_test, mu, cov_inv)

    plot_results(scores, y_test, save_dir=".")