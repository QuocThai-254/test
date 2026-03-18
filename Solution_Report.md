# Báo Cáo Giải Quyết Bài Toán Phân Loại Hình Ảnh (Image Classification)

## 1. Giới thiệu Bài Toán
Yêu cầu của bài toán là huấn luyện một mô hình AI để phân loại các hình ảnh vào 6 tag khác nhau tương ứng với 6 thư mục trong tập `data_train` (`em_bé_chơi_verified`, `ngày_tết_verified`, `other`, `thiennhien`, `trekking_verified`, `tụ_họp_verified`). 
Mô hình sau đó được đánh giá trên tập `data_test`. Thời gian thực hiện là trong 3 ngày và không được phép sử dụng các mô hình LLM hay Multi-modal.

## 2. Phân Tích Dữ Liệu & Hướng Tiếp Cận
Tập dữ liệu huấn luyện (`data_train`) có dung lượng khá nhỏ (khoảng 261 hình ảnh chia cho 6 lớp). Với lượng dữ liệu mỏng như vậy, việc huấn luyện một mô hình Deep Learning (ví dụ như ResNet hay VGG) từ đầu (from scratch) sẽ dẫn đến hiện tượng **Overfitting** rất nặng (mô hình học thuộc lòng tập train nhưng dự đoán kém trên tập test).

**Giải pháp:** 
Sử dụng phương pháp **Transfer Learning** (Học chuyển giao). Cụ thể, chọn mô hình **MobileNetV2** đã được pre-trained trên tập dữ liệu Imagenet. 
- **Lý do chọn MobileNetV2**: Đây là mô hình CNN rất nhẹ, tốc độ inference nhanh, phù hợp cho các bài toán phân loại cơ bản và thiết bị có cấu hình thấp.
- **Cách thức thực hiện**: Đóng băng (freeze) các layer trích xuất đặc trưng (feature extractor) của MobileNetV2, chỉ thay thế và huấn luyện lớp Classifier cuối cùng để phân loại thành 6 classes tương ứng với bài toán.

## 3. Quá Trình Huấn Luyện (Training)
Quá trình huấn luyện được trình bày chi tiết và có thể thực thi trong file `Image_Classification_Solution.ipynb`.

### 3.1. Data Augmentation (Tăng cường dữ liệu)
Để đối phó với hiện tượng thiếu dữ liệu, các kỹ thuật Augmentation được áp dụng trên tập training sử dụng `torchvision.transforms`:
- `Resize((224, 224))`: Đưa ảnh về kích thước chuẩn của MobileNetV2.
- `RandomHorizontalFlip()`: Lật ngang ảnh ngẫu nhiên.
- `RandomRotation(15)`: Xoay ảnh ngẫu nhiên tối đa 15 độ.
- `ColorJitter`: Thay đổi ngẫu nhiên độ sáng, độ tương phản, độ bão hòa màu để làm phong phú dữ liệu.
- `Normalize`: Chuẩn hóa vector ảnh theo tham số chuẩn của ImageNet.

### 3.2. Cấu hình Mô Hình & Hyperparameters
- **Model**: `models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)`
- **Loss Function**: `CrossEntropyLoss` (Hàm mất mát chuẩn cho Multi-class Classification).
- **Optimizer**: `Adam` với learning rate = 0.001 (Chỉ tối ưu hóa phần Classifier Head).
- **Batch Size**: 32.
- **Số lượng Epochs**: 10.

## 4. Quá Trình Đánh Giá (Evaluation) trên Tập Test
Sau khi train xong Model bằng file `Image_Classification_Solution.ipynb`, mô hình sẽ tự động chạy Inference trên toàn bộ hình ảnh có trong `data_test`.
Các Report đánh giá bao gồm:
1. **Classification Report**: Thể hiện các chỉ số quan trọng như `Precision`, `Recall`, `F1-Score` cho từng tag (từng lớp).
2. **Confusion Matrix**: Ma trận nhầm lẫn vẽ bằng Seaborn giúp trực quan hóa được lớp nào đang bị nhận diện nhầm với lớp nào nhiều nhất.

*(Bạn có thể thực thi file `.ipynb` được đính kèm để xem biểu đồ kết quả thực tế trên máy)*

## 5. Hướng Cải Thiện Cho Bài Toán (Future Improvements)
Để cải thiện độ chính xác và đưa mô hình này vào môi trường Production thực tế, có thể áp dụng các hướng sau:

1. **Gia tăng tập dữ liệu (Data Collection)**: 
   - Đây là yếu tố quan trọng cốt lõi. Cần thu thập thêm hàng ngàn hình ảnh đa dạng cho mỗi tag để mô hình học được nhiều pattern hơn.
   - Sử dụng các kỹ thuật tổng hợp dữ liệu (Synthetic data) nếu cần.
2. **Fine-Tuning Chuyên Sâu**:
   - Thay vì chỉ train lớp Classifier cuối, ta có thể "Unfreeze" (mở khóa) một số layer CNN cuối cùng của MobileNetV2 và train toàn bộ với Learning Rate rất nhỏ (ví dụ `1e-5`) để mô hình thích nghi sâu hơn với cấu trúc của tập ảnh này.
3. **Sử Dụng Mô Hình Phức Tạp Hơn**:
   - Nếu có tài nguyên GPU mạnh, có thể nâng cấp từ MobileNetV2 lên **ResNet50**, **EfficientNet-B2** hoặc **ConvNeXt** vì chúng có năng lực trích xuất đặc trưng tốt hơn mạnh mẽ.
4. **Xử Lý Mất Cân Bằng Dữ Liệu (Class Imbalance)**:
   - Nếu số lượng ảnh giữa lớp `other` quá chênh lệch so với `ngày_tết_verified`, ta cần cấu hình trọng số class-weights đưa vào `CrossEntropyLoss`, hoặc dùng kỹ thuật Oversampling/Undersampling.
5. **Hyperparameter Tuning**: 
   - Cài đặt thư viện `Optuna` hoặc `Ray Tune` để quét và tìm ra bộ tham số (Learning Rate, Batch Size, Dropout Rate) đem lại kết quả tối ưu nhất tự động.
