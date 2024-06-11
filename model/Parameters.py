from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

class Params():
    """Data"""
    base_path = '/Users/viviane/Desktop/Internship/Path_Flow_Prediction/Output/5by5_Data'

    """Training"""
    batch_size = 32
    train_size = 2000
    val_size = 400
    test_size = 500
    device = 'gpu'
    input_dim = 1162
    output_dim = 3
    d_model = 64
    heads = 8
    N = 2 # number of encoder/decoder layer
    epochs = 200
    lr = 0.001
    drop_out = 0.1
    reg_factor = 0.01
    loss_fn = MeanAbsoluteError()
