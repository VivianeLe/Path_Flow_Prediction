# %run tf_attention.py
# %run tf_helpers.py 
# %run parameters.py
import tensorflow as tf
from model.tf_helpers import *
from model.Parameters import *

files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
path_set_dict = path_encoder(files)

class Dataset:
    def __init__(self, files):
        # self.path_encoded = path_encoder()  # Get path encode dictionary
        self.X = []
        self.Y = []
        self.X_mask = []
        self.Y_mask = []

        for file_name in tqdm(files):
            # file_name = ''.join([BASE_PATH, str(start_from+i)])
            x, y, xMask, yMask = generate_xy(file_name, path_set_dict)
            self.X.append(x)
            self.Y.append(y)
            self.X_mask.append(xMask)
            self.Y_mask.append(yMask)
        
        self.X = tf.stack(self.X, axis=0)
        self.Y = tf.stack(self.Y, axis=0)
        self.X_mask = tf.stack(self.X_mask, axis=0)
        self.Y_mask = tf.stack(self.Y_mask, axis=0)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx]

    def to_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.X_mask, self.Y_mask))
        dataset = dataset.shuffle(buffer_size=len(self.X)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

def get_test_set(files):
    X = []
    Y = []
    X_Mask = []
    Y_Mask = []
    Scalers = []
    for file_name in tqdm(files) :
        x, y, xmask, ymask, scaler = generate_xy(file_name, path_set_dict, test_set=True)
        X.append(x)
        Y.append(y)
        X_Mask.append(xmask)
        Y_Mask.append(ymask)
        Scalers.append(scaler)
    X = tf.stack(X, axis=0)
    Y = tf.stack(Y, axis=0)
    X_Mask = tf.stack(X_Mask, axis=0)
    Y_Mask = tf.stack(Y_Mask, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y, X_Mask, Y_Mask))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset, Scalers