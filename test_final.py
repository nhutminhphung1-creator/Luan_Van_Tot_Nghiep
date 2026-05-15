import torch
from torch.utils.data import DataLoader
from dataset import TripletDataset
from model   import JointAnomalyDetector
from loss    import AnomalyLoss
# Đảm bảo file train.py của bạn có hàm eval_with_threshold và get_class_weights
from train   import eval_with_threshold, get_class_weights

def run_final_test():
    # 1. Cấu hình thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # 2. Nạp dữ liệu Test (những dữ liệu model chưa từng thấy)
    test_ds = TripletDataset("dataset/samples.json", split="test")
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)

    # 3. Khởi tạo Model và nạp "Bộ não" đã huấn luyện
    model = JointAnomalyDetector(embed_dim=128).to(device)
    try:
        ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("✅ Đã nạp model thành công từ checkpoints/best_model.pt")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file best_model.pt trong thư mục checkpoints!")
        return

    # 4. Khởi tạo Criterion (Cần thiết cho hàm eval)
    train_ds = TripletDataset("dataset/samples.json", split="train")
    cw = get_class_weights(train_ds).to(device)
    criterion = AnomalyLoss(lambda_contrast=0.5, class_weights=cw)

    # 5. Chạy đánh giá với Threshold tối ưu 0.40
    print("\n" + "="*30)
    print("── FINAL TEST RESULTS (Threshold=0.40) ──")
    print("="*30)
    
    res = eval_with_threshold(model, test_dl, criterion, device, threshold=0.40)
    
    # In báo cáo ra màn hình
    print(res["report"])
    print("="*30)
    print("Hoàn thành!")

if __name__ == "__main__":
    run_final_test()