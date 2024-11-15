# Hệ thống nhận diện loại da và gợi ý sản phẩm chăm sóc da

Dự án này sử dụng Flask và TensorFlow để xây dựng một ứng dụng web nhận diện loại da từ hình ảnh người dùng và gợi ý các sản phẩm chăm sóc da phù hợp. Mô hình học sâu (CNN) được huấn luyện để phân loại các loại da như Da Dầu, Da Khô, Da Mụn và Da Bình Thường. Sau khi nhận diện, hệ thống sẽ gợi ý các sản phẩm từ cơ sở dữ liệu phù hợp với loại da của người dùng.

## Mục lục
- [Yêu cầu](#yêu-cầu)
- [Cài đặt](#cài-đặt)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Huấn luyện mô hình](#huấn-luyện-mô-hình)
- [Tác giả](#tác-giả)

---

### Yêu cầu

Để chạy ứng dụng này, bạn cần cài đặt các công cụ và thư viện sau:
- Python 3.9 hoặc cao hơn
- Pip (Python package manager)
- TensorFlow 2.x
- Flask
- SQLite3

---

### Cài đặt

1. **Clone repository**:
    ```bash
    git clone <repository-url>
    cd my-flask-app
    ```

2. **Tạo virtual environment và cài đặt các thư viện**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên MacOS/Linux
    venv\Scripts\activate     # Trên Windows
    pip install -r requirements.txt
    ```

3. **Cấu hình cơ sở dữ liệu**:
    - Tạo cơ sở dữ liệu SQLite và bảng `products` với các trường `name`, `description`, `effect`, `skin_type`, `image` để lưu thông tin sản phẩm.
    - Bạn có thể tạo cơ sở dữ liệu bằng cách sử dụng SQLite3 hoặc công cụ quản lý cơ sở dữ liệu như DB Browser for SQLite.

4. **Chạy ứng dụng Flask**:
    ```bash
    python app.py
    ```

Ứng dụng sẽ chạy ở `http://localhost:5000`.

---

### Chạy ứng dụng

Ứng dụng cung cấp các tính năng chính:
1. **Nhận diện loại da**: Người dùng tải lên hình ảnh và hệ thống sẽ phân loại loại da và gợi ý các sản phẩm chăm sóc da phù hợp.
2. **Thêm sản phẩm**: Người quản trị có thể thêm sản phẩm vào cơ sở dữ liệu qua trang "Thêm sản phẩm".
3. **Xem sản phẩm**: Hiển thị danh sách tất cả các sản phẩm trong cơ sở dữ liệu.

Trang chính của ứng dụng sẽ cho phép người dùng tải lên hình ảnh và nhận được kết quả phân loại loại da và các sản phẩm phù hợp.

---

### Huấn luyện mô hình

Mô hình nhận diện loại da được huấn luyện bằng cách sử dụng dữ liệu ảnh. Các bước huấn luyện mô hình như sau:

1. **Cài đặt môi trường huấn luyện**:
    - Cài đặt TensorFlow và các thư viện cần thiết từ `requirements.txt`.

2. **Chuẩn bị dữ liệu**:
    - Đảm bảo bạn có bộ dữ liệu hình ảnh đã được chia thành các thư mục cho từng loại da (`train`, `validation`).
    
3. **Huấn luyện mô hình**:
    Chạy tệp `train_model.py` để huấn luyện mô hình CNN:

    ```bash
    python train_model.py
    ```

    Sau khi huấn luyện xong, mô hình sẽ được lưu vào tệp `my_model_v2.keras`.

---

### Tác giả

Dự án này được phát triển bởi Lưu Minh Thắng. 

Nếu có bất kỳ câu hỏi nào, vui lòng liên hệ qua email: thangluu.111104@gmail.com.

---

### Các tệp trong dự án:
- `app.py`: Tệp ứng dụng Flask chính.
- `train_model.py`: Tệp huấn luyện mô hình nhận diện loại da.
- `requirements.txt`: Các thư viện Python cần thiết cho dự án.
- `database.db`: Cơ sở dữ liệu SQLite lưu trữ thông tin sản phẩm.
- `model/my_model_v2.keras`: Mô hình học sâu (CNN) đã huấn luyện.
#   s k i n - c a r e - r e c o m m e n d a t i o n - s y s t e m 
 
 
