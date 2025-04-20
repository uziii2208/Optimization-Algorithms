# So sánh các thuật toán tối ưu trong Deep Learning

Dự án này thực hiện việc so sánh hiệu suất của các thuật toán tối ưu phổ biến (SGD, Adam, RMSprop) trên bộ dữ liệu MNIST để phân tích các đặc điểm và lựa chọn thuật toán phù hợp cho từng bài toán.

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/uziii2208/Optimization-Algorithms.git
cd [this project path]
```

2. Tạo môi trường ảo Python:
```bash
python -m venv venv
```

3. Kích hoạt môi trường ảo:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements/requirements.txt
```

## Cấu trúc dự án

```
.
├── data/                   # Thư mục chứa dữ liệu MNIST
├── notebooks/             
│   └── optimizer_comparison.ipynb  # Notebook chính để thực hiện thí nghiệm
├── requirements/
│   └── requirements.txt    # File chứa các dependency
├── results/                # Thư mục lưu kết quả thí nghiệm
└── src/
    ├── models.py          # Định nghĩa mô hình CNN
    ├── trainer.py         # Class ModelTrainer để huấn luyện mô hình
    └── utils.py           # Các hàm tiện ích
```

## Sử dụng

1. Khởi động Jupyter Lab:
```bash
jupyter lab
```

2. Mở notebook `notebooks/optimizer_comparison.ipynb`

3. Chạy các cell trong notebook để:
   - Tải và chuẩn bị dữ liệu MNIST
   - Huấn luyện mô hình với các optimizer khác nhau
   - Phân tích và so sánh kết quả

## Kết quả

Sau khi chạy thí nghiệm, bạn sẽ có:
- Đồ thị so sánh loss và accuracy của các optimizer
- Biểu đồ theo dõi learning rate và gradient norm
- File CSV chứa các metric cuối cùng
- Phân tích về tốc độ hội tụ của từng thuật toán

## Kết luận chính

1. **Tốc độ hội tụ**:
   - Adam thường hội tụ nhanh nhất
   - SGD với momentum hội tụ chậm hơn nhưng ổn định
   - RMSprop có tốc độ hội tụ trung bình

2. **Đề xuất sử dụng**:
   - Adam: Phù hợp cho hầu hết các bài toán, đặc biệt khi cần hội tụ nhanh
   - SGD với momentum: Phù hợp cho các bài toán cần độ ổn định cao
   - RMSprop: Tốt cho các mạng RNN và bài toán có gradient thay đổi nhiều