from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

class Params():
    """Data"""
    base_path = 'Output/5by5_Data'

    """Training"""
    batch_size = 32
    train_size = 4000
    val_size = 1000
    device = 'gpu'
    input_dim = 5
    output_dim = 3
    d_model = 64
    heads = 8
    N = 4 # number of encoder/decoder layer
    epochs = 100
    lr = 0.002
    drop_out = 0.1
    reg_factor = 0.01
    loss_fn = MeanAbsoluteError()
