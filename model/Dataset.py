from model.tf_helpers import *
from model.Parameters import *
from tqdm import tqdm

param = Params()

class Dataset:
    def __init__(self, size, start_from=0):
        self.path_encoded = path_encoder()  # Get path encode dictionary
        self.X = []
        self.Y = []
        # self.X_mask = []
        # self.Y_mask = []

        for i in tqdm(range(size)):
            file_name = ''.join([param.base_path, str(start_from+i)])
            x, y = generate_xy(file_name, self.path_encoded)
            self.X.append(x)
            self.Y.append(y)
            # self.X_mask.append(xMask)
            # self.Y_mask.append(yMask)
        
        self.X = tf.stack(self.X, axis=0)
        self.Y = tf.stack(self.Y, axis=0)
        # self.X_mask = tf.stack(self.X_mask, axis=0)
        # self.Y_mask = tf.stack(self.Y_mask, axis=0)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx]
        return self.X[idx], self.Y[idx]

    def to_tf_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        dataset = dataset.shuffle(buffer_size=len(self.X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset 