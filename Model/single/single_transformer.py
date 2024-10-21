import tensorflow as tf
from keras import layers as tfl
from keras import regularizers, Sequential
from tqdm.notebook import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.activations import sigmoid, linear, softmax

class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        # leaky_relu = LeakyReLU(alpha=0.5)
        self.ffn = Sequential([
            tfl.Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            tfl.Dropout(dropout),
            tfl.Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.dropout = tfl.Dropout(dropout)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        attn_output = self.attn_layer(query=x, key=x, value=x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        x = self.layer_norm1(self.dropout(x))

        ffn_output = self.ffn(x, training=training)
        x = self.layer_norm2(x + ffn_output)
        x = self.layer_norm2(self.dropout(x))
        return x

class Encoder(tfl.Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout, l2_reg):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]
        self.dense = tfl.Dense(3, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return self.dense(output)

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
        # leaky_relu = LeakyReLU(alpha=0.5)
        self.ffn = Sequential([
            tfl.Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            tfl.Dropout(dropout),
            tfl.Dense(output_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.layer_norm3 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)

    def call(self, x, encoder_output, training=None):
        attn1 = self.mha1(query=x, key=x, value=x, training=training)
        x = self.layer_norm1(x + self.dropout1(attn1))
        # x = self.layer_norm3(self.dropout3(x)) # ignore this layer, loss 0.0022 in 150 epochs

        attn2 = self.mha2(query=x, key=encoder_output, value=encoder_output, training=training)
        x = self.layer_norm2(x + self.dropout2(attn2))
        x = self.layer_norm3(self.dropout3(x))

        ffn_output = self.ffn(x, training=training)
        x = self.layer_norm3(x + ffn_output)
        x = self.layer_norm3(self.dropout3(x))
        return x

class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout, l2_reg):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]

    def call(self, x, encoder_output, training=None):
        output = x
        for layer in self.layers:
            output = layer(output, encoder_output, training=training)
        return output

class Transformer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, d_model, E_layer, D_layer, heads, dropout, l2_reg):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, E_layer, heads, dropout, l2_reg)
        self.decoder = Decoder(output_dim, d_model, D_layer, heads, dropout, l2_reg)
        self.activation = Activation('sigmoid')

    def call(self, x, y, training=None):
        encoder_output = self.encoder(x, training=training)
        decoder_output = self.decoder(y, encoder_output, training=training)
        decoder_output = self.activation(decoder_output)
        return decoder_output

    def eval(self):
        for layer in self.encoder.layers:
            layer.trainable = False
        for layer in self.decoder.layers:
            layer.trainable = False

    def train(self):
        for layer in self.encoder.layers:
            layer.trainable = True
        for layer in self.decoder.layers:
            layer.trainable = True

    def fit(self, train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device):
        # Define the early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )

        train_losses = []
        val_losses = []

        src, trg = next(iter(train_data_loader))
        self(src, trg, training=True)

        self.compile(optimizer=optimizer, loss=loss_fn)

        with tqdm(total=epochs, unit="epoch") as pbar:
            for epoch in range(epochs):
                # Training phase
                self.train()
                total_train_loss = 0
                for src, trg in train_data_loader:
                    with tf.device(device):
                        with tf.GradientTape() as tape:
                            output = self.call(src, trg)
                            loss = loss_fn(trg, output)

                        # Backpropagate and update the model
                        gradients = tape.gradient(loss, self.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                        total_train_loss += loss.numpy()
                        # pbar.set_description(f"Train Loss: {total_train_loss / len(train_data_loader):.4f}")

                # Validation phase
                self.eval()
                total_val_loss = 0
                for src, trg in val_data_loader:
                    with tf.device(device):
                        output = self.call(src, trg)
                        loss = loss_fn(trg, output)
                        total_val_loss += loss.numpy()

                        # pbar.set_description(f"Val Loss: {total_val_loss / len(val_data_loader):.4f}")

                pbar.update(1)
                train_losses.append(total_train_loss / len(train_data_loader))
                val_losses.append(total_val_loss / len(val_data_loader))
                print(f"Epoch: {epoch+1} - Train Loss: {total_train_loss/len(train_data_loader):.4f}, Val Loss: {total_val_loss/len(val_data_loader):.4f}")

                # Check for early stopping
                if early_stopping.model is not None:
                    early_stopping.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                    reduce_lr.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                    if early_stopping.stopped_epoch > 0:
                        print(f"Early stopping triggered at epoch {early_stopping.stopped_epoch + 1}")
                        break
        return self, train_losses, val_losses

    def predict(self, x):
        encoder_output = self.encoder(x, training=False)
        decoder_input = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], 3))
        predictions = self.decoder(decoder_input, encoder_output, training=False)
        predictions = self.activation(predictions)
        return predictions

def inversed(normed, scaler):
    # normed: 625x3
    tensor = scaler.inverse_transform(np.transpose(normed))
    tensor = np.transpose(tensor)
    return tensor

import time
def predict_withScaler(model, test_data_loader, scalers, device):
    model.eval()
    predicted_values = []
    scaler_idx = 0
    for src, trg in test_data_loader:
        with tf.device(device):
            # output = model.predict(src, src_mask, tgt_mask)
            start = time.time()
            output = model.call(src, trg)
            end = time.time()
            print("Finish predicting in: ", (end-start)/len(src), "seconds/data point")
            for i in range(len(src)):
                scaler = scalers[scaler_idx]
                scaler_idx +=1
                # pred_matrix = inversed(output[i].numpy(), scaler) # reverse transform by each row
                pred_matrix = scaler.inverse_transform(output[i].numpy()) # inverse transform by column
                predicted_values.append(pred_matrix)

    return predicted_values

def predict_percentage(model, test_data_loader, device):
    model.eval()
    predicted_values = []
    for src, trg in test_data_loader:
        with tf.device(device):
            output = model.call(src, trg)
            for i in range(len(src)):
                predicted_values.append(output[i].numpy())

    return predicted_values