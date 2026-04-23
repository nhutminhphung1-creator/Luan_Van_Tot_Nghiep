import gpxpy
import pandas as pd
import os

def convert_gpx_to_csv(folder_path):
    # Lấy danh sách tất cả file .gpx trong thư mục
    files = [f for f in os.listdir(folder_path) if f.endswith('.gpx')]
    
    if not files:
        print("Không tìm thấy file .gpx nào!")
        return

    for filename in files:
        gpx_path = os.path.join(folder_path, filename)
        
        try:
            with open(gpx_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        # Trích xuất các thông tin quan trọng
                        data.append({
                            'timestamp': point.time.timestamp() if point.time else None,
                            'time_readable': point.time,
                            'lat': point.latitude,
                            'lon': point.longitude,
                            'speed': getattr(point, 'speed', 0) # Lấy vận tốc nếu có
                        })
            
            # Tạo DataFrame và lưu thành CSV
            df = pd.DataFrame(data)
            csv_name = filename.replace('.gpx', '.csv')
            df.to_csv(os.path.join(folder_path, csv_name), index=False)
            print(f"✅ Đã chuyển đổi: {filename} -> {csv_name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file {filename}: {e}")

# Chạy hàm chuyển đổi tại thư mục hiện tại
if __name__ == "__main__":
    current_folder = os.getcwd() # Lấy thư mục đang mở trong VS Code
    convert_gpx_to_csv(current_folder)