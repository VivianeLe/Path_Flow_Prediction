# CHANGE THE DIRECTORY OF DATA YOU WANT TO TRAIN
FOLDERS = '../Solution/SiouxFalls/Output1'
TEST_FILES_2 = '../Solution/SiouxFalls/Remove2'
TEST_FILES_3 = '../Solution/SiouxFalls/Remove3'
UNIQUE_PATH_DICT = '../Generate_data/SiouxFalls/unique_paths.pkl'
NODE_POSITION = '../Generate_data/SiouxFalls/SiouxFalls_node.csv'

# CHART TITLES
TRAIN_HISTORY_TITLE = "SINGLE CLASS NETWORK - TRAINING HISTORY"
ERROR_TITLE = "SINGLE CLASS NETWORK"

DATA_SIZE = 3000
TRAIN_RATE = 0.7
VAL_RATE = 0.2
TEST_RATE = 0.1
BATCH_SIZE = 64

# TRAINING 
device = 'gpu'
input_dim = 6
output_dim = 3
d_model = 128
heads = 8
E_layer = 8
D_layer = 1
epochs = 400
learning_rate = 0.001
dropout=0.1
l2_reg=1e-6