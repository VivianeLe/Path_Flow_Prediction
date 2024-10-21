from single.single_helpers import * 
from single.single_transformer import *
from single.single_params import *
from single.single_dataset import *
from plotting import *
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.losses import MeanSquaredError
from time import time
import concurrent.futures

# files = load_files_from_folders(FOLDERS, max_files=100)
# path_set_dict = path_encoder(files)
unique_set = read_file(UNIQUE_PATH_DICT)

def load_data(files, unique_set):
    dataset = Dataset(files, unique_set)
    data_loader = dataset.to_tf_dataset(BATCH_SIZE)
    return data_loader

def predict_and_plot(model, TEST_FILES, unique_set, link_miss=2):
    test_files = load_files_from_folders(TEST_FILES, max_files=int(DATA_SIZE*TEST_RATE))
    test_data_loader, scalers = get_test_set(test_files, unique_set)

    # PREDICTING 
    print("Start predicting...")
    pred_tensor = predict_withScaler(model, test_data_loader, scalers, device)

    # Calculate error
    print_result_single(pred_tensor, test_files, NODE_POSITION, f"{ERROR_TITLE} - Missing {link_miss} links")

def main():
    # LOAD DATA
    files = load_files_from_folders(FOLDERS, max_files=DATA_SIZE)
    train_files, val_files, test_files = split_dataset(files, TRAIN_RATE, VAL_RATE)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_train = executor.submit(load_data, train_files, unique_set)
        future_val = executor.submit(load_data, val_files, unique_set)
        
        # Get the results
        train_data_loader = future_train.result()
        val_data_loader = future_val.result()

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

    predict_and_plot(model, TEST_FILES_2, unique_set, 2)
    predict_and_plot(model, TEST_FILES_3, unique_set, 3)

    plt.show()

if __name__ == "__main__":
    main()
