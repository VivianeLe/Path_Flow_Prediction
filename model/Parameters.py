from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

class Params():
    """Data"""
    base_path = 'C:/Users/Vu Tuan Minh/Desktop/VA/Path_Flow_Prediction/Output/5by5_Data'

    """Training"""
    batch_size = 8
    train_size = 2000
    val_size = 400
    test_size = 200
    device = 'cuda'
    sequence_length = 625
    input_dim = 7
    output_dim = 3
    d_model = 64
    heads = 8
    N = 2 # number of encoder/decoder layer
    epochs = 100
    lr = 0.001
    drop_out = 0.1
    reg_factor = 0.01
    loss_fn = MeanAbsoluteError()
