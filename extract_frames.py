import cv2
import pandas as pd
from pathlib import Path
from datetime import timedelta

def extract_frames_by_timestamp(
    video_path: str,
    gps_df: pd.DataFrame,
    output_dir: str,
    sizes: list = [512, 256]
):
    """
    Trích frame từ video tại mỗi giây, match với GPS timestamp.
    Tên file đặt theo timestamp để sync sau.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)          # thường ~30fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps       # giây

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy thời điểm bắt đầu từ GPS (dòng đầu tiên)
    t_start = gps_df['time_readable'].iloc[0]

    saved = 0
    for sec in range(int(video_duration)):
        # Seek đến đúng giây thứ `sec`
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Tính timestamp tương ứng
        t_frame = t_start + timedelta(seconds=sec)
        ts_str = t_frame.strftime("%Y%m%d_%H%M%S")

        for size in sizes:
            resized = cv2.resize(frame, (size, size))
            fname = out_dir / f"{ts_str}_{size}.jpg"
            cv2.imwrite(str(fname), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved += 1

    cap.release()
    print(f"✅ Saved {saved} frames from {Path(video_path).name}")
    return saved

if __name__ == "__main__":
    from parse_gps import load_gps

    # Map tên video → CSV tương ứng (dựa trên naming convention của bạn)
    # "Nha_truong2" → "20260416-084608 - Nha_truong2.csv"
    pairs = [
        ("video/Nha_truong2.mp4",  "gps/gpx/Nha_truong2.csv"),
        ("video/Nha_truong3.mp4",  "gps/gpx/Nha_truong3.csv"),
        ("video/Nha_truong4.mp4",  "gps/gpx/Nha_truong4.csv"),
        ("video/Nha_truong5.mp4",  "gps/gpx/Nha_truong5.csv"),
        ("video/Nha_truong_loivongxp.mp4",  "gps/gpx/Nha_truong_loivongxp.csv"),

        ("video/Truong_nha2.mp4",  "gps/gpx/Truong_nha2.csv"),
        ("video/Truong_nha3.mp4",  "gps/gpx/Truong_nha3.csv"),
        ("video/Truong_nha4.mp4",  "gps/gpx/Truong_nha4.csv"),
        ("video/Truong_nha5.mp4",  "gps/gpx/Truong_nha5.csv"),
        ("video/Truong_nha_loivihe.mp4",  "gps/gpx/Truong_nha_loivihe.csv"),

    ]

    for video_path, csv_path in pairs:
        gps_df = load_gps(csv_path)
        trip_name = Path(video_path).stem
        extract_frames_by_timestamp(
            video_path=video_path,
            gps_df=gps_df,
            output_dir=f"dataset/frames/{trip_name}",
            sizes=[512, 256]
        )