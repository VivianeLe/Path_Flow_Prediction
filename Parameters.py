from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

data_size = 2000
class Params():
    """Data"""
    base_path = 'Output/5by5_Data'

    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)
    test_size = data_size - train_size - val_size
    batch_size = 4
    gradient_accumulation_steps = 4
    device = 'gpu'
    input_dim = 5
    output_dim = 3
    d_model = 512
    heads = 8
    E_layer = 8
    D_layer = 1
    epochs = 100
    lr = 0.002
    dropout = 0.5 # for dense layer
    mha_dropout = 0.1 # multihead attention dropout
    l2_reg = 1e-4 # for kernel_regularizer
    loss_fn = MeanSquaredError()