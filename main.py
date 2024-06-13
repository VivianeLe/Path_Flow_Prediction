from model.tf_helpers import *
from model.attention_nomask import *
from model.Parameters import *
from model.Dataset import Dataset
from tensorflow.keras.optimizers import Adam
import time

param = Params()

def create_dataloader():
    train_dataset = Dataset(param.train_size)
    train_data_loader = train_dataset.to_tf_dataset(param.batch_size)

    val_dataset = Dataset(param.val_size, start_from=param.train_size)
    val_data_loader = val_dataset.to_tf_dataset(param.batch_size)

    test_dataset = Dataset(param.test_size, start_from=param.train_size+param.val_size)
    test_dataloader = test_dataset.to_tf_dataset(param.batch_size)
    return train_data_loader, val_data_loader, test_dataloader

def main():
    # Create dataloader
    train_data_loader, val_data_loader, test_dataloader = create_dataloader()
    model = Transformer(input_dim=param.input_dim, output_dim=param.output_dim, 
                        d_model=param.d_model, N=param.N, heads=param.heads, 
                        dropout=param.drop_out)
    optimizer = Adam(learning_rate=param.lr)

    start = time.time()
    trained_model, train_losses, val_losses = training_loop(model, train_data_loader, val_data_loader, param.epochs, param.loss_fn, optimizer, param.device)
    train_time = time.time()-start
    # model.compile(optimizer=optimizer, loss=param.loss_fn)

    plot_loss(train_losses, val_losses, param.epochs, param.lr, train_time, param.N, param.d_model)
    rmse, mae, mape = evaluate_model(trained_model, test_dataloader, param.device)
    print(f"RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

if __name__ == "__main__":
    main()