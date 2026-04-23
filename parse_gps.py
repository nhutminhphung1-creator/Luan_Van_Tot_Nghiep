import pandas as pd
import numpy as np
from pathlib import Path

def load_gps(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Đảm bảo đúng tên cột: timestamp, time_readable, lat, lon, speed
    df['time_readable'] = pd.to_datetime(df['time_readable'], utc=True)
    df = df.sort_values('time_readable').reset_index(drop=True)

    # Lọc nhiễu: tốc độ âm hoặc > 120 km/h
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]

    # Tính bearing (hướng di chuyển) giữa 2 điểm liên tiếp
    df['bearing'] = compute_bearing(df['lat'], df['lon'])
    return df

def compute_bearing(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Tính góc hướng di chuyển (0–360°) giữa các điểm liên tiếp."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r.diff()
    dlon = lon_r.diff()
    x = np.sin(dlon) * np.cos(lat_r)
    y = np.cos(lat_r.shift()) * np.sin(lat_r) - \
        np.sin(lat_r.shift()) * np.cos(lat_r) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return bearing.fillna(0)

if __name__ == "__main__":
    csv_dir = Path("gps/gpx")
    for csv_file in csv_dir.glob("*.csv"):
        df = load_gps(str(csv_file))
        print(f"{csv_file.name}: {len(df)} rows")
        print(df[['time_readable','lat','lon','speed','bearing']].head(3))
        print()