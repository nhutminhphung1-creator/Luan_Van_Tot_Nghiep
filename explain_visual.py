"""
GradCAM cho Swin Transformer — visualize vùng ảnh mà model chú ý.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
import json, cv2
from pathlib import Path

from model   import JointAnomalyDetector


class SwinGradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        

    def _save_activation(self, module, input, output):
        self.activations = output.detach()   # [B*3, num_patches, C]

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, gps, imgs_512, imgs_256, target_class=1):
        # 1. Đảm bảo input có requires_grad
        imgs_512 = imgs_512.clone().requires_grad_(True)
        
        # 2. Forward pass
        # Chế độ eval nhưng vẫn giữ gradient cho input
        self.model.zero_grad()
        outputs = self.model(gps, imgs_512, imgs_256)
        
        # 3. Lấy activations từ biến đã lưu trong model.py
        # acts lúc này đang là [B, 3, 49, 768]
        acts = self.model.visual_encoder.last_feat_512
        B, T, P, C = acts.shape
        
        # Vì ta cần gradient tại acts, ta phải gọi retain_grad() 
        # TRƯỚC KHI gọi backward
        acts.retain_grad()
        
        # 4. Backward pass (Chỉ gọi 1 lần duy nhất)
        score = outputs["logits"][:, target_class].sum()
        score.backward()

        # 5. Tính toán CAM
        # grads: [B, 3, 49, 768]
        grads = acts.grad
        
        # Chuyển về dạng [B*3, P, C] để tính toán cho từng ảnh trong triplet
        acts_flat = acts.view(B*T, P, C)
        grads_flat = grads.view(B*T, P, C)

        # Global Average Pooling cho Gradients để làm trọng số (Weights)
        weights = grads_flat.mean(dim=1, keepdim=True) # [B*3, 1, 768]
        
        # Tính Weighted Sum của activations
        cam = torch.sum(weights * acts_flat, dim=-1)   # [B*3, 49]

        return torch.relu(cam) # Chỉ lấy các giá trị dương (ReLU)


def cam_to_heatmap(cam_1d, size=224):
    """
    Chuyển đặc trưng 1D của Swin Transformer về dạng Heatmap 2D.
    Swin-Tiny thường trả về 49 patches (lưới 7x7).
    """
    # Tính toán n dựa trên kích thước thực tế (49 -> n=7)
    # Thêm .detach() trước khi gọi .numpy()
    total_elements = cam_1d.shape[0]
    n = int(np.sqrt(total_elements)) 
    
    if n * n != total_elements:
        raise ValueError(f"Không thể reshape vector size {total_elements} thành hình vuông!")

    # Sửa dòng này:
    hm = cam_1d.detach().cpu().reshape(n, n).numpy()
    
    # Chuẩn hóa giá trị về 0-1
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    
    # Resize từ 7x7 lên 224x224 để đè lên ảnh gốc
    return cv2.resize(hm, (size, size))


def overlay(img_tensor, heatmap, alpha=0.45):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.permute(1,2,0).numpy()
    img  = (img * std + mean).clip(0,1)
    color = plt.cm.jet(heatmap)[:,:,:3]
    return (alpha * color + (1-alpha) * img).clip(0,1)


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


def visualize_sample(model, sample, device, save_path, gradcam):
    keys = ["P_start", "P_mid", "P_end"]

    # Load ảnh
    raw, f512, f256 = [], [], []
    for k in keys:
        img = Image.open(sample["images"][k]["img_512"]).convert("RGB")
        raw.append(img.resize((224,224)))
        f512.append(TF(img))
        f256.append(TF(Image.open(sample["images"][k]["img_256"]).convert("RGB")))

    imgs_512 = torch.stack(f512).unsqueeze(0).to(device)
    imgs_256 = torch.stack(f256).unsqueeze(0).to(device)

    # GPS vector
    feats = []
    for k in keys:
        p = sample[k]
        feats += [p["lat"], p["lon"], p["velocity"], p["bearing"]]
    m = sample["meta"]
    feats += [m["deviation_ratio"], m["avg_speed"], float(m["sudden_stop"])]
    gps = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)

    # GradCAM
    model.eval()
    cam = gradcam.generate(gps, imgs_512, imgs_256, target_class=1)

    # Predict
    with torch.no_grad():
        out  = model(gps, imgs_512, imgs_256)
        prob = torch.softmax(out["logits"], dim=1)[0,1].item()
    pred = "ABNORMAL" if prob >= 0.40 else "NORMAL"

    # ── Figure: 3 × (original + heatmap) + info panel ──
    fig = plt.figure(figsize=(16, 7), facecolor='#0d0d1a')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)
    clr = ["#42A5F5", "#FFA726", "#EF5350"]
    lbl = ["P_start", "P_mid", "P_end"]

    for t in range(3):
        ax_o = fig.add_subplot(gs[0, t])
        ax_o.imshow(raw[t])
        spd = sample[lbl[t]]["velocity"]
        ax_o.set_title(f"{lbl[t]}\nv = {spd:.1f} km/h",
                       color=clr[t], fontsize=9, fontweight='bold')
        ax_o.axis("off")

        ax_h = fig.add_subplot(gs[1, t])
        hm      = cam_to_heatmap(cam[t].cpu())
        ov_img  = overlay(f512[t].cpu(), hm)
        ax_h.imshow(ov_img)
        ax_h.set_title("GradCAM ↑ Vùng model chú ý",
                       fontsize=8, color='#aaa')
        ax_h.axis("off")

    # Info panel
    ax_i = fig.add_subplot(gs[:, 3])
    ax_i.set_facecolor('#1a1a2e')
    ax_i.axis("off")
    gt_str   = " ABNORMAL" if sample["label"] == 1 else " NORMAL"
    pred_str = " ABNORMAL" if pred == "ABNORMAL" else " NORMAL"
    correct  = " True" if (sample["label"]==1) == (pred=="ABNORMAL") else " False"
    info = (
        f"Ground Truth : {gt_str}\n"
        f"Prediction   : {pred_str}\n"
        f"Result    : {correct}\n"
        f"Prob Abnormal: {prob:.3f}\n\n"
        f"── GPS Metrics ────────\n"
        f"Deviation ratio : {m['deviation_ratio']:.3f}\n"
        f"Avg speed       : {m['avg_speed']:.1f} km/h\n"
        f"Sudden stop     : {m['sudden_stop']}\n\n"
        f"── Tọa độ ─────────────\n"
        f"Start ({sample['P_start']['lat']:.5f},\n"
        f"       {sample['P_start']['lon']:.5f})\n"
        f"Mid   ({sample['P_mid']['lat']:.5f},\n"
        f"       {sample['P_mid']['lon']:.5f})\n"
        f"End   ({sample['P_end']['lat']:.5f},\n"
        f"       {sample['P_end']['lon']:.5f})\n"
    )
    ax_i.text(0.05, 0.97, info, transform=ax_i.transAxes,
              fontsize=8.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#12122a', alpha=0.95),
              color='white')

    title_color = "#EF5350" if pred == "ABNORMAL" else "#66BB6A"
    fig.suptitle(f"Explainability — {pred}  |  P(abnormal) = {prob:.3f}",
                 fontsize=13, color=title_color, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    print(f"  ✅ {Path(save_path).name}  [{pred}  p={prob:.3f}]")