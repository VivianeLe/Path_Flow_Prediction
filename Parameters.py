from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

class Params():
    """Data"""
    base_path = 'C:/Users/Vu Tuan Minh/Desktop/VA/Path_Flow_Prediction/Output/5by5_Data'

    """Training"""
    batch_size = 8
    train_size = 2000
    val_size = 400
    test_size = 200
    device = 'gpu'
    sequence_length = 625
    input_dim = 7
    output_dim = 3
    d_model = 60
    heads = 6
    N = 4 # number of encoder/decoder layer
    epochs = 100
    lr = 0.001
    drop_out = 0.5
    reg_factor = 0.01
    loss_fn = MeanAbsoluteError()
