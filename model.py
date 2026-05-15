import torch
import torch.nn as nn
import timm

# ══════════════════════════════════════════════════════
# NHÁNH 1: Trajectory Encoder (GPS → z_traj)
# ══════════════════════════════════════════════════════
class TrajectoryEncoder(nn.Module):
    """
    Input : [B, 15]  — vector GPS đã chuẩn hóa
    Output: [B, 128] — z_traj

    Vector 15 chiều gồm:
      - P_start: lat, lon, velocity, bearing  (4 feat)
      - P_mid  : lat, lon, velocity, bearing  (4 feat)
      - P_end  : lat, lon, velocity, bearing  (4 feat)
      - meta   : deviation_ratio, avg_speed, sudden_stop (3 feat)
    
    Reshape → [B, 3, 5]:
      step 0: [lat, lon, vel, bearing, deviation_ratio]
      step 1: [lat, lon, vel, bearing, avg_speed]
      step 2: [lat, lon, vel, bearing, sudden_stop]
    """
    def __init__(self, input_dim=15, hidden=128, nhead=4, num_layers=2):
        super().__init__()
        self.token_dim = 5  # mỗi time-step có 5 feature

        self.input_proj = nn.Linear(self.token_dim, hidden)
        self.pos_emb    = nn.Embedding(3, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=256, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 15]
        B = x.size(0)

        # Tách GPS features: 3 điểm × 4 feat = 12, meta = 3
        gps  = x[:, :12].view(B, 3, 4)   # [B, 3, 4]
        meta = x[:, 12:]                  # [B, 3]

        # Ghép meta vào từng time-step (mỗi step nhận 1 meta feature)
        tokens = torch.cat([
            gps,
            meta.unsqueeze(-1)            # [B, 3, 1]
        ], dim=-1)                        # [B, 3, 5]

        # Project + positional encoding
        h = self.input_proj(tokens)                                    # [B, 3, hidden]
        pos = self.pos_emb(torch.arange(3, device=x.device))          # [3, hidden]
        h = h + pos                                                    # [B, 3, hidden]

        # Transformer
        h = self.transformer(h)           # [B, 3, hidden]

        # Flatten 3 token → pool
        z = self.pool(h.reshape(B, -1))   # [B, hidden]
        return z


# ══════════════════════════════════════════════════════
# NHÁNH 2: Visual Encoder (Ảnh → z_vis)
# ══════════════════════════════════════════════════════
class VisualEncoder(nn.Module):
    """
    Input : imgs_512 [B, 3, 3, 512, 512]
            imgs_256 [B, 3, 3, 256, 256]
    Output: z_vis [B, 128]

    - Dùng Swin-T (nhẹ hơn SwinV2-B, phù hợp dataset ~2857 mẫu)
    - Shared weights cho cả 3 frame (start, mid, end)
    - Multi-scale fusion: concat feat 512 + feat 256 → project
    """
    def __init__(self, embed_dim=128):
        super().__init__()

        # Swin-Tiny pretrained — đủ mạnh, không overfit với dataset nhỏ
        self.backbone_512 = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True, num_classes=0,  # num_classes=0 → lấy feature
            img_size=224                      # resize về 224 trong transform
        )
        self.backbone_256 = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True, num_classes=0,
            img_size=224
        )
        feat_dim = self.backbone_512.num_features  # 768 với Swin-T

        # Multi-scale fusion: (feat_512 + feat_256) × 3 frames → embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )

    def _encode_frames(self, backbone, imgs):
        B, T, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * T, C, H, W)

        # ❗ QUAN TRỌNG: dùng forward_features LẤY PATCH TOKENS
        feats = backbone.forward_features(imgs_flat)  # [B*3, 49, 768]

        return feats.view(B, T, 49, -1)  # [B, 3, 49, 768]

    def forward(self, imgs_512, imgs_256):
        # Resize 512/256 → 224 (Swin-T yêu cầu 224)
        # (đã resize trong transform của Dataset, hoặc dùng F.interpolate)
        import torch.nn.functional as F
        B, T, C, H, W = imgs_512.shape
        if H != 224:
            imgs_512 = F.interpolate(
                imgs_512.view(B*T, C, H, W), size=224, mode='bilinear'
            ).view(B, T, C, 224, 224)
        if imgs_256.shape[-1] != 224:
            B2, T2, C2, H2, W2 = imgs_256.shape
            imgs_256 = F.interpolate(
                imgs_256.view(B2*T2, C2, H2, W2), size=224, mode='bilinear'
            ).view(B, T, C2, 224, 224)

        # Encode từng scale
        feat_512 = self._encode_frames(self.backbone_512, imgs_512)  # [B,3,49,768]
        feat_256 = self._encode_frames(self.backbone_256, imgs_256)  # [B,3,49,768]

        # 🔥 GIỮ LẠI PATCH CHO GRADCAM
        self.last_feat_512 = feat_512   # lưu lại để GradCAM dùng

        # 🔧 POOL để giữ pipeline cũ
        feat_512 = feat_512.mean(dim=2)  # [B,3,768]
        feat_256 = feat_256.mean(dim=2)  # [B,3,768]

        # Multi-scale concat theo frame, rồi flatten
        fused = torch.cat([feat_512, feat_256], dim=-1)   # [B, 3, 1536]
        z_vis = self.fusion(fused.reshape(B, -1))          # [B, 128]
        return z_vis


# ══════════════════════════════════════════════════════
# NHÁNH 3: Cross-Modal Fusion + Classifier
# ══════════════════════════════════════════════════════
class JointAnomalyDetector(nn.Module):
    """
    Ghép z_traj + z_vis → dự đoán Normal/Abnormal
    """
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()

        self.traj_encoder  = TrajectoryEncoder(input_dim=15, hidden=embed_dim)
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim)

        # Cross-modal attention: z_traj attend vào z_vis
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4,
            dropout=0.1, batch_first=True
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),  # z_traj + z_vis + z_cross
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
        )

        # Classifier
        self.classifier = nn.Linear(64, num_classes)

        # Projection heads cho Contrastive Loss
        self.proj_traj = nn.Linear(embed_dim, 64)
        self.proj_vis  = nn.Linear(embed_dim, 64)

    def forward(self, gps, imgs_512, imgs_256):
        # Encode
        z_traj = self.traj_encoder(gps)                       # [B, 128]
        z_vis  = self.visual_encoder(imgs_512, imgs_256)      # [B, 128]

        # Cross-modal attention: GPS "hỏi" ảnh
        z_q = z_traj.unsqueeze(1)   # [B, 1, 128] — query
        z_kv = z_vis.unsqueeze(1)   # [B, 1, 128] — key/value
        z_cross, _ = self.cross_attn(z_q, z_kv, z_kv)
        z_cross = z_cross.squeeze(1)  # [B, 128]

        # Concat tất cả → classify
        z_all = torch.cat([z_traj, z_vis, z_cross], dim=-1)  # [B, 384]
        feat  = self.fusion(z_all)                            # [B, 64]
        logits = self.classifier(feat)                        # [B, 2]

        # Projection cho contrastive loss (dùng khi training)
        p_traj = self.proj_traj(z_traj)  # [B, 64]
        p_vis  = self.proj_vis(z_vis)    # [B, 64]

        return {
            "logits":  logits,
            "z_traj":  z_traj,
            "z_vis":   z_vis,
            "p_traj":  p_traj,
            "p_vis":   p_vis,
            "feat":    feat,
        }