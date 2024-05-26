# Path_Flow_Prediction

Ver 3: 
X shape: 625, 7
X mask: 625, 1
Y shape: 625, 3
Y mask: 625, 1

Ver 4: 
X shape: 25, 175 
X mask: 1, 175
Y shape: 25, 75 
Y mask: 1, 75
Trong attention: X và Y đổi thành 175, 25 và 75, 25, apply mask. Sau đó đổi lại về shape ban đầu để đưa vào MHA2
MHA2 trong decoder ko sử dụng mask.

Ver 5: 
Khởi tạo 3 tensor Wq, Wk, Wv shape input_dim, d_model
Nhân X (shape 25, 175) với Wq, Wk, Wv => Q, K, V shape 25, d_model
Nhân Q với K transpose, chia dk**0.5 => attention tensor 25, 25
Không có mask, thay thế các giá trị = 0 trong input tensor = -inf
Nhân attention tensor với V => 25, d_model
