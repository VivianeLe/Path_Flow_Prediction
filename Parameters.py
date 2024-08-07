from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

DATA_SIZE = 600
FOLDERS = [f'Solution/Output{i+1}' for i in range(4)]
TRAIN_RATE = 0.7
VAL_RATE = 0.2
TEST_RATE = 0.1
BATCH_SIZE = 8

class Params():
    """Data"""
    base_path = 'Solution/Output1/5by5_Data'

    # train_size = int(data_size * 0.8)
    # val_size = int(data_size * 0.1)
    # test_size = data_size - train_size - val_size
    gradient_accumulation_steps = 4
    device = 'gpu'
    input_dim = 7
    output_dim = 3
    d_model = 512
    heads = 8
    E_layer = 8
    D_layer = 2
    epochs = 100
    lr = 0.001
    dropout = 0.5 # for dense layer
    mha_dropout = 0.1 # multihead attention dropout
    l2_reg = 1e-4 # for kernel_regularizer
    loss_fn = MeanSquaredError()