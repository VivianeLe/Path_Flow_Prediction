# Path_Flow_Prediction

Ver 2:
Use built-in MHA class
X shape: 625, 1165 (chuyen thanh 625, 7 vi ko can adj matrix)
Y shape: 625, 3
No mask

Ver 3: 
X shape: 625, 7
X mask: 625, 1
Y shape: 625, 3
Y mask: 625, 1
trong attention: đổi chiều mask thành 1, 625 trước khi apply

Ver 4: 
Apply mask in self-define MHA 
Normalize Y before training, apply sigmoid activate function
Run well with scaled output 
Apply mask on predicted output to reduce error when evaluate 

<<<<<<< HEAD
=======
Ver 5: 
Khởi tạo 3 tensor Wq, Wk, Wv shape input_dim, d_model
Nhân X (shape 25, 175) với Wq, Wk, Wv => Q, K, V shape 25, d_model
Nhân Q với K transpose, chia dk**0.5 => attention tensor 25, 25
Không có mask (hoặc thử có mask), thay thế các giá trị = 0 trong input tensor = -inf
Nhân attention tensor với V => 25, d_model
* Thử apply label encoder với mỗi path set của mỗi OD pair (mỗi set of 3 là 1 số)
>>>>>>> origin/main
