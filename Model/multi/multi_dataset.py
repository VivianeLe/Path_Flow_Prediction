from multi.multi_helpers import *
from multi.multi_params import *

class Dataset:
    def __init__(self, files, unique_set):
        # self.path_encoded = path_encoder()  # Get path encode dictionary
        self.X = []
        self.Y_c, self.Y_t = [], []

        for file_name in tqdm(files):
            x, y_c = generate_xy(file_name, unique_set)
            self.X.append(x)
            self.Y_c.append(y_c)

        self.X = tf.stack(self.X, axis=0)
        self.Y_c = tf.stack(self.Y_c, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_c[idx]

    def to_tf_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y_c))
        dataset = dataset.shuffle(buffer_size=len(self.X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

def get_test_set(files, unique_set):
    X = []
    Y = []
    Scaler = []
    for file_name in tqdm(files) :
        x, y, scaler = generate_xy(file_name, unique_set, test_set=True)
        X.append(x)
        Y.append(y)
        Scaler.append(scaler)

    X = tf.stack(X, axis=0)
    Y = tf.stack(Y, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset, Scaler