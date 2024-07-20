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

Folder Generate_data includes:
    - code to generate random OD matrix (4 files of OD matrix then combine them into 1 file)
    - read the OD matrix and network file (.tntp) to find k shortest paths
    - use Gurobipy to solve UE problem.

Folder UE Solution includes:
    - Output1: solution for network of full 25 nodes, 80 links
    - Output2: solution for network of 25 nodes, 75 links (random remove 5 links)
    - Output3: solution for network of 25 nodes, 70 links (random remove 10 links)