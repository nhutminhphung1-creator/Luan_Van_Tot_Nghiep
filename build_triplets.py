import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Bước 1: Hàm tiện ích ────────────────────────────────────────────────────

def load_gps(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['time_readable'] = pd.to_datetime(df['time_readable'], utc=True)
    df = df.sort_values('time_readable').reset_index(drop=True)
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)].reset_index(drop=True)

    # Tính bearing
    lat_r = np.radians(df['lat'])
    lon_r = np.radians(df['lon'])
    dlon  = lon_r.diff()
    x = np.sin(dlon) * np.cos(lat_r)
    y = (np.cos(lat_r.shift()) * np.sin(lat_r)
         - np.sin(lat_r.shift()) * np.cos(lat_r) * np.cos(dlon))
    df['bearing'] = ((np.degrees(np.arctan2(x, y)) + 360) % 360).fillna(0)
    return df

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ─── Bước 2: Tạo triplets ────────────────────────────────────────────────────

def build_triplets(gps_df: pd.DataFrame, window: int = 10) -> list:
    triplets = []
    n    = len(gps_df)
    step = window // 2

    for i in range(0, n - window, step):
        s, m, e = gps_df.iloc[i], gps_df.iloc[i + window//2], gps_df.iloc[i + window]

        d_actual   = (haversine(s.lat,s.lon, m.lat,m.lon)
                    + haversine(m.lat,m.lon, e.lat,e.lon))
        d_straight = haversine(s.lat, s.lon, e.lat, e.lon)
        dev_ratio  = d_actual / (d_straight + 1e-6)

        speed_win  = gps_df.iloc[i:i+window]['speed'].values
        sudden_stop = bool(np.sum(speed_win < 1.0) > 5)

        label = 1 if (dev_ratio > 1.5 or sudden_stop) else 0

        triplets.append({
            "id": f"{i:06d}",
            "P_start": {"lat": float(s.lat), "lon": float(s.lon),
                        "timestamp": str(s.time_readable),
                        "velocity": float(s.speed), "bearing": float(s.bearing)},
            "P_mid":   {"lat": float(m.lat), "lon": float(m.lon),
                        "timestamp": str(m.time_readable),
                        "velocity": float(m.speed), "bearing": float(m.bearing)},
            "P_end":   {"lat": float(e.lat), "lon": float(e.lon),
                        "timestamp": str(e.time_readable),
                        "velocity": float(e.speed), "bearing": float(e.bearing)},
            "meta": {
                "deviation_ratio": float(dev_ratio),
                "avg_speed":       float(np.mean(speed_win)),
                "sudden_stop":     sudden_stop,
            },
            "label": label,
        })
    return triplets

# ─── Bước 3: Sync ảnh ────────────────────────────────────────────────────────

def list_images(frames_dir: Path, size: int):
    imgs = sorted(frames_dir.glob(f"*_{size}.jpg"))
    return imgs

def find_nearest_image(frames_dir: Path, ts: pd.Timestamp, size: int):
    ts_local = ts.tz_convert("Asia/Ho_Chi_Minh")

    candidates = list(frames_dir.glob(f"*_{size}.jpg"))
    if not candidates:
        return None

    def ts_diff(p: Path):
        try:
            stem = p.stem  # "20260416_014608_512"
            parts = stem.rsplit("_", 1)
            t = pd.Timestamp(parts[0], tz="Asia/Ho_Chi_Minh")
            return abs((t - ts_local).total_seconds())
        except Exception:
            return 9999

    best = min(candidates, key=ts_diff)
    return str(best) if ts_diff(best) <= 3 else None   # tăng tolerance lên 3s


def sync_images(triplets: list, frames_dir: str) -> list:
    fdir = Path(frames_dir)

    frames_512 = list_images(fdir, 512)
    frames_256 = list_images(fdir, 256)

    # ❗ Nếu không có ảnh → skip luôn
    if not frames_512 or not frames_256:
        print(f"⚠️  Không tìm thấy ảnh trong {frames_dir}")
        return []

    synced = []

    for idx, t in enumerate(triplets):
        images = {}

        # ── TRY 1: match theo timestamp ──
        success = True
        for key in ["P_start", "P_mid", "P_end"]:
            ts = pd.Timestamp(t[key]["timestamp"])

            img512 = find_nearest_image(fdir, ts, 512)
            img256 = find_nearest_image(fdir, ts, 256)

            if not img512 or not img256:
                success = False
                break

            images[key] = {
                "img_512": img512,
                "img_256": img256
            }

        # ── TRY 2: fallback theo index (QUAN TRỌNG) ──
        if not success:
            if idx + 2 >= len(frames_512):
                continue

            images = {
                "P_start": {
                    "img_512": str(frames_512[idx]),
                    "img_256": str(frames_256[idx]),
                },
                "P_mid": {
                    "img_512": str(frames_512[idx+1]),
                    "img_256": str(frames_256[idx+1]),
                },
                "P_end": {
                    "img_512": str(frames_512[idx+2]),
                    "img_256": str(frames_256[idx+2]),
                }
            }

        t["images"] = images
        synced.append(t)

    return synced

# ─── Bước 4: Main — map đúng tên folder của bạn ──────────────────────────────

# Mỗi phần tử: (đường dẫn CSV, thư mục frames tương ứng, nhãn trip)
PAIRS = [
    # ── Nha truong (đường đến trường) ──
    ("gps/gpx/Nha_truong2.csv",   "dataset/frames/Nha_truong2"),
    ("gps/gpx/Nha_truong3.csv",   "dataset/frames/Nha_truong3"),
    ("gps/gpx/Nha_truong4.csv",   "dataset/frames/Nha_truong4"),
    ("gps/gpx/Nha_truong5.csv",   "dataset/frames/Nha_truong5"),
    ("gps/gpx/Nha_truong8.csv",   "dataset/frames/Nha_truong8"),
    # ── Truong nha (đường về nhà) ──
    ("gps/gpx/Truong_nha2.csv",   "dataset/frames/Truong_nha2"),
    ("gps/gpx/Truong_nha3.csv",   "dataset/frames/Truong_nha3"),
    ("gps/gpx/Truong_nha4.csv",   "dataset/frames/Truong_nha4"),
    ("gps/gpx/Truong_nha5.csv",   "dataset/frames/Truong_nha5"),
    ("gps/gpx/Truong_nha8.csv",   "dataset/frames/Truong_nha8"),
    # ── Lối vòng / Lối vi he (abnormal candidates) ──
    ("gps/gpx/Nha_truong_loivongxp.csv",  "dataset/frames/Nha_truong_loivongxp"),
    ("gps/gpx/Truong_nha_loivihe.csv",  "dataset/frames/Truong_nha_loivihe"),
    ("gps/gpx/Nha_truong_duongdai.csv",  "dataset/frames/Nha_truong_duongdai"),
    ("gps/gpx/Truong_nha_hem.csv",  "dataset/frames/Truong_nha_hem"),
]

if __name__ == "__main__":
    Path("dataset").mkdir(exist_ok=True)
    all_samples = []
    stats = {"normal": 0, "abnormal": 0, "skipped_trips": []}

    for csv_path, frames_dir in PAIRS:
        csv_p = Path(csv_path)
        frm_p = Path(frames_dir)

        # Bỏ qua nếu thiếu file
        if not csv_p.exists():
            print(f"⚠️  CSV không tồn tại: {csv_p}")
            stats["skipped_trips"].append(str(csv_p))
            continue
        if not frm_p.exists():
            print(f"⚠️  Frames folder không tồn tại: {frm_p}")
            stats["skipped_trips"].append(str(frm_p))
            continue

        gps_df   = load_gps(str(csv_p))
        triplets = build_triplets(gps_df, window=10)
        synced   = sync_images(triplets, frames_dir)

        n_trip   = sum(s["label"] == 0 for s in synced)
        a_trip   = sum(s["label"] == 1 for s in synced)
        stats["normal"]   += n_trip
        stats["abnormal"] += a_trip

        # Gán trip_id để trace sau này
        trip_id = frm_p.name
        for s in synced:
            s["trip_id"] = trip_id

        all_samples.extend(synced)
        print(f"✅ {csv_p.name:<45} → {len(synced):>4} samples  "
              f"(N={n_trip}, A={a_trip})")

    # Lưu
    out_path = Path("dataset/samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"""
╔══════════════════════════════════════╗
║         BUILD DATASET XONG           ║
╠══════════════════════════════════════╣
║  Tổng samples  : {len(all_samples):<6}               ║
║  Normal  (0)   : {stats['normal']:<6}               ║
║  Abnormal (1)  : {stats['abnormal']:<6}               ║
║  Saved         : dataset/samples.json ║
╚══════════════════════════════════════╝""")

    if stats["skipped_trips"]:
        print(f"\n⚠️  Bị bỏ qua: {stats['skipped_trips']}")