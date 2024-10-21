from single.single_helpers import * 
from single.single_transformer import *
from single.single_params import *
from single.single_dataset import *
from plotting import *
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.losses import MeanSquaredError
from time import time

# files = load_files_from_folders(FOLDERS, max_files=100)
# path_set_dict = path_encoder(files)
unique_set = read_file(UNIQUE_PATH_DICT)

def main():
    # LOAD DATA
    files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
    train_files, val_files, test_files = split_dataset(files, TRAIN_RATE, VAL_RATE)

    train_dataset = Dataset(train_files, unique_set)
    train_data_loader = train_dataset.to_tf_dataset(BATCH_SIZE)

    val_dataset = Dataset(val_files, unique_set)
    val_data_loader = val_dataset.to_tf_dataset(BATCH_SIZE)


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

    # if use trained model with a network of randomly removing 2 links
    test_files = load_files_from_folders(TEST_FILES_2, max_files=int(DATA_SIZE*0.1))
    test_data_loader, scalers = get_test_set(test_files, unique_set)

    # PREDICTING 
    print("Start predicting...")
    pred_tensor = predict_withScaler(model, test_data_loader, scalers, device)

    # Calculate error
    print_result_single(pred_tensor, test_files, NODE_POSITION, f"{ERROR_TITLE} - Missing 2 links")

    # REMOVE 3 LINKS
    test_files = load_files_from_folders(TEST_FILES_3, max_files=int(DATA_SIZE*0.1))
    test_data_loader, scalers = get_test_set(test_files, unique_set)

    # PREDICTING 
    print("Start predicting...")
    pred_tensor = predict_withScaler(model, test_data_loader, scalers, device)

    # Calculate error
    print_result_single(pred_tensor, test_files, NODE_POSITION, f"{ERROR_TITLE} - Missing 3 links")

    plt.show()

if __name__ == "__main__":
    main()
