# Hướng dẫn chi tiết thực hiện thí nghiệm so sánh Optimizers

Tài liệu này sẽ hướng dẫn bạn từng bước thực hiện các thí nghiệm so sánh các thuật toán tối ưu (SGD, Adam, RMSprop) trên Jupyter Notebook.

## Chuẩn bị môi trường

1. **Kích hoạt môi trường ảo và khởi động Jupyter Lab**:
   ```bash
   # Kích hoạt môi trường
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   
   # Khởi động Jupyter Lab
   jupyter lab
   ```

2. **Mở notebook**:
   - Trong giao diện Jupyter Lab, điều hướng đến thư mục `notebooks`
   - Mở file `optimizer_comparison.ipynb`

## Thực hiện thí nghiệm

### 1. Thiết lập và Import (Cell 1)
```python
import sys
sys.path.append('../src')

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from models import SimpleCNN
from trainer import ModelTrainer
from utils import plot_training_curves, save_results_to_csv, compare_convergence_speed

# Đặt random seed để đảm bảo tính tái lập
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

### 2. Chuẩn bị dữ liệu (Cell 2)
```python
# Định nghĩa transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Tải dữ liệu MNIST
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
```

### 3. Huấn luyện với các Optimizer (Cell 3)
```python
# Thực hiện huấn luyện với các optimizer khác nhau
optimizers = ['SGD', 'Adam', 'RMSprop']
learning_rates = [0.01, 0.001, 0.0001]
results = {}

for opt_name in optimizers:
    for lr in learning_rates:
        print(f'\nTraining with {opt_name}, learning rate = {lr}')
        key = f'{opt_name}_lr_{lr}'
        results[key] = train_with_optimizer(opt_name, learning_rate=lr)
```

### 4. Phân tích kết quả (Cell 4)
```python
# Vẽ đồ thị so sánh
plot_training_curves(results)
plt.savefig('../results/optimizer_comparison.png')
plt.show()

# Lưu kết quả vào file CSV
results_df = save_results_to_csv(results)
display(results_df)

# So sánh tốc độ hội tụ
convergence_epochs = compare_convergence_speed(results, target_accuracy=95.0)
print('\nSố epoch cần để đạt độ chính xác 95%:')
for opt_name, epochs in convergence_epochs.items():
    print(f'{opt_name}: {epochs if epochs != float("inf") else "Không đạt target"} epochs')
```

## Giải thích các tham số và kết quả

### 1. Hyperparameters
- **Learning rates**: Thử nghiệm với 3 giá trị (0.01, 0.001, 0.0001)
  - 0.01: Learning rate lớn, có thể hội tụ nhanh nhưng không ổn định
  - 0.001: Learning rate trung bình, cân bằng giữa tốc độ và ổn định
  - 0.0001: Learning rate nhỏ, ổn định nhưng hội tụ chậm

- **Batch size**: 64
  - Giá trị này cân bằng giữa tốc độ huấn luyện và memory usage
  - Có thể điều chỉnh tùy theo GPU/RAM có sẵn

### 2. Phân tích biểu đồ
- **Loss Curves**:
  - Đường càng dốc → tốc độ học càng nhanh
  - Đường không ổn định → learning rate có thể quá cao
  - Đường ngang → model có thể đã hội tụ hoặc bị mắc kẹt

- **Accuracy Curves**:
  - Gap lớn giữa train và val → overfitting
  - Accuracy không tăng → underfitting hoặc learning rate không phù hợp

- **Learning Rate Changes**:
  - Adam và RMSprop sẽ có learning rate thay đổi theo thời gian
  - SGD giữ learning rate cố định (trừ khi dùng scheduler)

- **Gradient Norm**:
  - Gradient lớn → không ổn định
  - Gradient quá nhỏ → vanishing gradient

### 3. So sánh các Optimizer

#### SGD với Momentum
- **Ưu điểm**:
  - Ổn định và generalization tốt
  - Ít hyperparameters cần tinh chỉnh
- **Nhược điểm**:
  - Hội tụ chậm hơn
  - Nhạy cảm với learning rate

#### Adam
- **Ưu điểm**:
  - Hội tụ nhanh
  - Tự điều chỉnh learning rate
- **Nhược điểm**:
  - Có thể overfit
  - Nhiều hyperparameters hơn

#### RMSprop
- **Ưu điểm**:
  - Xử lý tốt gradient không ổn định
  - Phù hợp với RNN/LSTM
- **Nhược điểm**:
  - Có thể kém ổn định hơn Adam
  - Cần tinh chỉnh hyperparameters cẩn thận

## Tips và Troubleshooting

1. **Nếu loss không giảm**:
   - Giảm learning rate
   - Kiểm tra dữ liệu và preprocessing
   - Thử optimizer khác

2. **Nếu accuracy thấp**:
   - Tăng số epochs
   - Điều chỉnh model architecture
   - Thử các hyperparameters khác

3. **Nếu gặp lỗi out of memory**:
   - Giảm batch size
   - Giảm model size
   - Sử dụng gradient accumulation

4. **Để có kết quả tốt nhất**:
   - Thử nhiều learning rates
   - So sánh với và không có momentum
   - Theo dõi validation metrics
   - Lưu các checkpoint tốt nhất

## Lưu ý quan trọng

1. **Tính nhất quán**: 
   - Sử dụng cùng random seed
   - Giữ các hyperparameters không đổi khi so sánh

2. **Lưu trữ kết quả**:
   - Đặt tên file có ý nghĩa
   - Ghi chú đầy đủ về setup và các thay đổi

3. **Phân tích sâu**:
   - So sánh không chỉ accuracy cuối cùng
   - Xem xét tốc độ hội tụ
   - Đánh giá độ ổn định

4. **Kết luận**:
   - Dựa trên nhiều metrics
   - Xem xét trade-off giữa tốc độ và độ chính xác
   - Đưa ra recommendation rõ ràng