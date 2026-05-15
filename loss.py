import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss với Cluster-Guided Negative Mining.
    
    Ý tưởng CM-CGNS:
    - Các mẫu cùng label (kể cả cùng nhóm Abnormal) → KHÔNG đẩy ra xa nhau
    - Chỉ đẩy xa các mẫu khác label
    - Kết quả: model học đặc trưng CỐT LÕI của "bất thường"
      thay vì học sự khác biệt bề mặt
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, proj_traj, proj_vis, labels):
        """
        proj_traj, proj_vis: [B, 64] — đã L2 normalize
        labels: [B] — 0 hoặc 1
        """
        B = labels.size(0)

        # L2 normalize
        z1 = F.normalize(proj_traj, dim=-1)
        z2 = F.normalize(proj_vis,  dim=-1)

        # Concat cả 2 modality → [2B, 64]
        z = torch.cat([z1, z2], dim=0)
        labels_2x = torch.cat([labels, labels], dim=0)  # [2B]

        # Similarity matrix [2B, 2B]
        sim = torch.matmul(z, z.T) / self.temp

        # Mask: positive pair = cùng label (kể cả cross-modal)
        label_mat = labels_2x.unsqueeze(0) == labels_2x.unsqueeze(1)  # [2B, 2B]
        # Loại bỏ diagonal (so sánh với chính nó)
        eye = torch.eye(2 * B, device=z.device).bool()
        label_mat = label_mat & ~eye

        # Cluster-guided: không dùng mẫu cùng cluster Abnormal làm negative
        # → đã handled bởi label_mat (cùng label = không phải negative)

        # Tính loss: log-softmax trên positive pairs
        exp_sim = torch.exp(sim)
        # Loại diagonal
        exp_sim = exp_sim * (~eye).float()

        # Với mỗi anchor i: loss = -log(sum_positives / sum_all_except_self)
        pos_sum = (exp_sim * label_mat.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # Tránh log(0)
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        # Chỉ tính loss với anchor có ít nhất 1 positive
        has_pos = label_mat.any(dim=1)
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        return loss[has_pos].mean()


class AnomalyLoss(nn.Module):
    """
    Loss tổng hợp = CrossEntropy + λ × ContrastiveLoss
    
    Lý do dùng cả hai:
    - CE loss: dạy model predict đúng nhãn
    - Contrastive loss: dạy model học không gian embedding có cấu trúc
      → giúp tổng quát hóa tốt hơn với mẫu chưa thấy
    """
    def __init__(self, lambda_contrast=0.3, class_weights=None, temperature=0.07):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.contrast = SupervisedContrastiveLoss(temperature)
        self.lam = lambda_contrast

    def forward(self, outputs, labels):
        logits  = outputs["logits"]
        p_traj  = outputs["p_traj"]
        p_vis   = outputs["p_vis"]

        loss_ce       = self.ce(logits, labels)
        loss_contrast = self.contrast(p_traj, p_vis, labels)

        total = loss_ce + self.lam * loss_contrast
        return {
            "total":    total,
            "ce":       loss_ce,
            "contrast": loss_contrast
        }