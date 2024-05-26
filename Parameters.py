from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

class Params():
    """Data"""
    base_path = 'Output/5by5_Data'

    """Training"""
    batch_size = 32
    train_size = 800
    val_size = 200
    device = 'gpu'
    input_dim = 25
    output_dim = 25
    d_model = 128
    N = 2 # number of encoder/decoder layer
    epochs = 100
    lr = 0.001
    # loss_fn = MeanSquaredError()
    loss_fn = MeanAbsoluteError()
