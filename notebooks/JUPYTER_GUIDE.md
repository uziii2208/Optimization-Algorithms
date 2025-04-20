# Hướng dẫn sử dụng Jupyter Lab cho thí nghiệm Optimizers

## 1. Khởi động Jupyter Lab

### Bước 1: Mở Terminal
- Mở Command Prompt hoặc PowerShell
- Di chuyển đến thư mục dự án:
```bash
cd e:\Cybersecurity\Algorithms
```
- Kích hoạt môi trường ảo:
```bash
.\venv\Scripts\activate
```
- Khởi động Jupyter Lab:
```bash
jupyter lab
```

### Bước 2: Truy cập Jupyter Lab
- Jupyter Lab sẽ tự động mở trong trình duyệt mặc định
- URL thường là: http://localhost:8888/lab

## 2. Giao diện Jupyter Lab

### 2.1. Thanh công cụ bên trái
- **File Browser**: Duyệt và quản lý files
- **Running Terminals and Kernels**: Xem các notebook đang chạy
- **Extension Manager**: Quản lý các extension
- **Table of Contents**: Xem cấu trúc notebook

### 2.2. Khu vực làm việc chính
- **File Browser**: Hiển thị cấu trúc thư mục
- **Notebook Editor**: Nơi bạn viết và chạy code
- **Terminal**: Thực thi lệnh shell
- **Text Editor**: Chỉnh sửa file văn bản

## 3. Làm việc với Notebook

### 3.1. Mở Notebook
1. Trong File Browser, điều hướng đến thư mục `notebooks`
2. Double-click vào file `optimizer_comparison.ipynb`

### 3.2. Các thao tác cơ bản với Cell

#### Chế độ Command Mode (phím tắt: Esc)
- **A**: Thêm cell mới phía trên
- **B**: Thêm cell mới phía dưới
- **X**: Cắt cell đang chọn
- **C**: Copy cell
- **V**: Paste cell
- **DD**: Xóa cell đang chọn
- **Z**: Undo xóa cell
- **M**: Chuyển cell sang Markdown
- **Y**: Chuyển cell sang Code

#### Chế độ Edit Mode (phím tắt: Enter)
- **Shift + Enter**: Chạy cell hiện tại và di chuyển xuống cell tiếp theo
- **Ctrl + Enter**: Chạy cell hiện tại và giữ nguyên vị trí
- **Alt + Enter**: Chạy cell hiện tại và tạo cell mới bên dưới

### 3.3. Thực thi Code
1. **Chọn Kernel**:
   - Click vào nút Kernel trên thanh công cụ
   - Chọn "Change Kernel" > Python 3

2. **Chạy từng Cell**:
   - Click vào cell cần chạy
   - Nhấn Shift + Enter hoặc click nút "Run" (▶️)

3. **Chạy tất cả Cells**:
   - Menu "Run" > "Run All Cells"
   - Hoặc Ctrl + Alt + R

### 3.4. Theo dõi Output

#### Cell Output
- Kết quả xuất hiện ngay dưới cell
- Các biểu đồ được hiển thị tự động
- Có thể xóa output bằng cách:
  - Click chuột phải vào cell > Clear Outputs
  - Menu "Edit" > "Clear All Outputs"

#### Progress Bar
- Thanh tiến trình hiển thị khi chạy vòng lặp với tqdm
- Hiển thị % hoàn thành và thời gian ước tính

## 4. Tips khi thực hiện thí nghiệm

### 4.1. Quản lý Notebook
- **Lưu thường xuyên**: Ctrl + S
- **Checkpoint**: File > Save and Checkpoint
- **Xuất notebook**: File > Export Notebook As > HTML/PDF

### 4.2. Debug và Xử lý lỗi
1. **Kiểm tra biến**:
```python
print(variable_name)
# hoặc
from pprint import pprint
pprint(complex_variable)
```

2. **Xem thông tin chi tiết về lỗi**:
```python
import traceback
try:
    # your code
except Exception as e:
    print(traceback.format_exc())
```

3. **Dừng huấn luyện**:
- Nhấn nút "Stop" (■) trên thanh công cụ
- Hoặc Kernel > Interrupt Kernel

### 4.3. Trực quan hóa kết quả
1. **Hiển thị nhiều biểu đồ**:
```python
plt.figure(figsize=(15, 5))
# Vẽ biểu đồ
plt.show()
```

2. **Lưu biểu đồ**:
```python
plt.savefig('../results/plot_name.png', dpi=300, bbox_inches='tight')
```

3. **Tương tác với biểu đồ**:
- Zoom: Click và kéo để phóng to vùng quan tâm
- Pan: Giữ Space + Click để di chuyển
- Reset: Home button trên thanh công cụ biểu đồ

### 4.4. Tối ưu hóa Workflow
1. **Auto-reload modules**:
```python
%load_ext autoreload
%autoreload 2
```

2. **Đo thời gian thực thi**:
```python
%%time
# code to time
```

3. **Hiển thị tất cả biến**:
```python
%whos
```

## 5. Xử lý các vấn đề thường gặp

### 5.1. Kernel bị treo
1. Kernel > Interrupt Kernel (⏹️)
2. Nếu không hiệu quả: Kernel > Restart Kernel (⟳)
3. Trường hợp xấu nhất: Kernel > Restart Kernel and Clear All Outputs

### 5.2. Out of Memory
1. Giải phóng bộ nhớ:
```python
import gc
gc.collect()
```
2. Giảm batch_size
3. Khởi động lại Kernel

### 5.3. Package không tìm thấy
1. Cài đặt package trong cell:
```python
!pip install package_name
```
2. Khởi động lại Kernel sau khi cài đặt

## 6. Best Practices

1. **Tổ chức Notebook**:
   - Đặt tên cell rõ ràng bằng markdown
   - Nhóm các cell liên quan
   - Thêm comments giải thích code

2. **Version Control**:
   - Clear tất cả outputs trước khi commit
   - Tạo checkpoint thường xuyên
   - Backup notebook quan trọng

3. **Tài nguyên**:
   - Theo dõi GPU usage (nếu có)
   - Đóng các notebook không sử dụng
   - Restart kernel định kỳ để giải phóng bộ nhớ

## 7. Các phím tắt hữu ích

### Command Mode (Esc)
- **H**: Hiển thị bảng phím tắt
- **Ctrl + Shift + P**: Mở Command Palette
- **L**: Hiển thị số dòng
- **O**: Ẩn/hiện output
- **Shift + M**: Merge cells đã chọn
- **Shift + L**: Bật/tắt số dòng

### Edit Mode (Enter)
- **Tab**: Code completion
- **Shift + Tab**: Hiển thị documentation
- **Ctrl + /** : Comment/uncomment
- **Ctrl + ]** : Indent
- **Ctrl + [** : Unindent