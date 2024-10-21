FOLDERS = '../Solution/SiouxFalls/MultiClass'
NODE_POSITION = '../Generate_data/SiouxFalls/SiouxFalls_node.csv'
UNIQUE_PATH_DICT = '../Generate_data/SiouxFalls/unique_paths.pkl'

# CHART TITLES
TRAIN_HISTORY_TITLE = "MULTI-CLASS NETWORK - TRAINING HISTORY"
CAR_ERROR_TITLE = "MULTI-CLASS NETWORK - CAR"
TRUCK_ERROR_TITLE = "MULTI-CLASS NETWORK - TRUCK"

# DATA SIZE
DATA_SIZE = 2000
TRAIN_RATE = 0.7 
VAL_RATE = 0.2
TEST_RATE = 0.1
BATCH_SIZE = 64

# TRAINING 
device = 'gpu'
input_dim = 8
output_dim = 6
d_model = 128
heads=8
E_layer = 8
D_layer = 2
epochs = 400
learning_rate = 0.001
dropout=0.1
l2_reg=1e-6