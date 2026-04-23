import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TripletDataset(Dataset):
    def __init__(self, json_path: str, split: str = "train"):
        with open(json_path) as f:
            all_samples = json.load(f)

        # Train/val/test split 70/15/15
        n = len(all_samples)
        idx = list(range(n))
        splits = {"train": idx[:int(n*0.7)],
                  "val":   idx[int(n*0.7):int(n*0.85)],
                  "test":  idx[int(n*0.85):]}
        self.samples = [all_samples[i] for i in splits[split]]

        # Transform cho ảnh
        self.tf_512 = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        self.tf_256 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path, tf):
        img = Image.open(path).convert("RGB")
        return tf(img)

    def _gps_vector(self, sample) -> torch.Tensor:
        """
        Chuyển GPS triplet → vector 15 chiều:
        [lat, lon, speed, bearing] × 3 điểm + [deviation_ratio, avg_speed, sudden_stop]
        """
        feats = []
        for key in ["P_start", "P_mid", "P_end"]:
            p = sample[key]
            feats += [p["lat"], p["lon"], p["velocity"], p["bearing"]]
        meta = sample["meta"]
        feats += [meta["deviation_ratio"], meta["avg_speed"],
                  float(meta["sudden_stop"])]
        return torch.tensor(feats, dtype=torch.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # GPS features (15-dim vector)
        gps_vec = self._gps_vector(s)  # shape: [15]

        # Images: 3 thời điểm × 2 kích thước
        imgs_512 = torch.stack([
            self._load_img(s["images"][k]["img_512"], self.tf_512)
            for k in ["P_start", "P_mid", "P_end"]
        ])  # shape: [3, 3, 512, 512]

        imgs_256 = torch.stack([
            self._load_img(s["images"][k]["img_256"], self.tf_256)
            for k in ["P_start", "P_mid", "P_end"]
        ])  # shape: [3, 3, 256, 256]

        label = torch.tensor(s["label"], dtype=torch.long)

        return {
            "gps": gps_vec,
            "imgs_512": imgs_512,
            "imgs_256": imgs_256,
            "label": label,
            "id": s["id"]
        }

if __name__ == "__main__":
    ds = TripletDataset("dataset/samples.json", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    print("GPS shape  :", batch["gps"].shape)      # [4, 15]
    print("Img512 shape:", batch["imgs_512"].shape) # [4, 3, 3, 512, 512]
    print("Label shape :", batch["label"].shape)    # [4]
    print("✅ Dataset OK!")