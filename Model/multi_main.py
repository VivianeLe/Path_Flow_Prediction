from multi.multi_helpers import * 
from multi.multi_transformer import *
from multi.multi_params import *
from multi.multi_dataset import *
from plotting import *
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.losses import MeanSquaredError
from time import time

# files = load_files_from_folders(FOLDERS, max_files=100)
# path_set_dict = path_encoder(files)
unique_set = read_file(UNIQUE_PATH_DICT)
# print("Path set number: ", len(unique_set))

def main():
    # LOAD DATA
    files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
    train_files, val_files, test_files1 = split_dataset(files, TRAIN_RATE, VAL_RATE)

    train_dataset = Dataset(train_files, unique_set)
    train_data_loader = train_dataset.to_tf_dataset(BATCH_SIZE)

    val_dataset = Dataset(val_files, unique_set)
    val_data_loader = train_dataset.to_tf_dataset(BATCH_SIZE)

    test_data_loader, Scalers = get_test_set(test_files1, unique_set)

    # TRAIN MODEL
    model = Transformer(input_dim=input_dim, output_dim=output_dim,
                        d_model=d_model, E_layer=E_layer, D_layer=D_layer,
                        heads=heads, dropout=dropout, l2_reg=l2_reg)
    loss_fn = MeanSquaredError()
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0, decay=l2_reg)
    start = time()
    model, train_loss, val_loss = model.fit(train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device)
    end = time()
    print("Finish training in: ", round((end-start)/3600, 2), "hours")

    # PLOTTING LOSS
    plot_loss(train_loss, val_loss, epochs, TRAIN_HISTORY_TITLE)

    # PREDICTING 
    print("Start predicting...")
    predicted_c, predicted_t = predict_withScaler(model, test_data_loader, Scalers, device)

    # Calculate error for car 
    print_result(predicted_c, test_files1, 'c', NODE_POSITION, CAR_ERROR_TITLE)

    # Calculate error for truck
    print_result(predicted_t, test_files1, 't', NODE_POSITION, TRUCK_ERROR_TITLE)

    plt.show()

if __name__ == "__main__":
    main()
