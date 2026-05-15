"""Chạy toàn bộ Explainability — GradCAM + Mahalanobis."""
import json, torch
from torch.utils.data import DataLoader
from pathlib import Path

from model        import JointAnomalyDetector
from dataset      import TripletDataset
from explain_visual import visualize_sample, SwinGradCAM
from explain_gps    import (extract_embeddings, fit_normal_distribution,
                             compute_scores, plot_results)

# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = JointAnomalyDetector(embed_dim=128).to(device)
ckpt  = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("✅ Model loaded\n")

Path("explain_output").mkdir(exist_ok=True)

# ── PHẦN A: GradCAM ───────────────────────────────────────────────────────────
print("═══ PHẦN A: GradCAM Visualization ═══")
with open("dataset/samples.json") as f:
    all_samples = json.load(f)

# Lấy mẫu từ test split (15% cuối)
n        = len(all_samples)
test_idx = list(range(int(n * 0.85), n))
test_smp = [all_samples[i] for i in test_idx]

# Chọn 4 Abnormal + 2 Normal để visualize
abnormal = [s for s in test_smp if s["label"] == 1][:4]
normal   = [s for s in test_smp if s["label"] == 0][:2]

gradcam = SwinGradCAM(model)

for i, s in enumerate(abnormal):
    visualize_sample(model, s, device,
                     f"explain_output/abnormal_{i+1}.png", gradcam)

for i, s in enumerate(normal):
    visualize_sample(model, s, device,
                     f"explain_output/normal_{i+1}.png", gradcam)

# ── PHẦN B: Mahalanobis ───────────────────────────────────────────────────────
print("\n═══ PHẦN B: Mahalanobis GPS Score ═══")
train_dl = DataLoader(TripletDataset("dataset/samples.json","train"),
                      batch_size=16, shuffle=False, num_workers=0)
test_dl  = DataLoader(TripletDataset("dataset/samples.json","test"),
                      batch_size=16, shuffle=False, num_workers=0)

print("Fitting distribution on train Normal samples...")
z_tr, y_tr = extract_embeddings(model, train_dl, device)
mu, cov_inv = fit_normal_distribution(z_tr, y_tr)

print("Scoring test samples...")
z_te, y_te = extract_embeddings(model, test_dl, device)
scores     = compute_scores(z_te, mu, cov_inv)
plot_results(scores, y_te, save_dir="explain_output")

# ── Tóm tắt ──────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════╗
║         EXPLAINABILITY HOÀN THÀNH            ║
╠══════════════════════════════════════════════╣
║  explain_output/abnormal_1~4.png             ║
║  explain_output/normal_1~2.png               ║
║  explain_output/mahalanobis_analysis.png     ║
╚══════════════════════════════════════════════╝
""")