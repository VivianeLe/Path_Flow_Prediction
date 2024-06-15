from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import math

data_size = 7000
class Params():
    """Data"""
    base_path = 'Output/5by5_Data'

    """Training"""
    batch_size = 8
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    test_size = data_size - train_size - val_size
    device = 'gpu'
    input_dim = 7
    output_dim = 3
    d_model = 48
    heads = 6
    N = 4 # number of encoder/decoder layer
    epochs = 100
    lr = 0.002
    dropout = 0.5
    # loss_fn = MeanSquaredError()
    loss_fn = MeanAbsoluteError()
