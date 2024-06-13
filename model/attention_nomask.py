import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import layers as tfl
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.mixed_precision import set_global_policy, Policy
policy = Policy('mixed_float16')
set_global_policy(policy)

class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_model, activation='relu')
        self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.dense2 = tfl.Dense(input_dim)
        self.dropout = tfl.Dropout(dropout)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
    def call(self, x, training=None):
        attn_output = self.attn_layer(query=x, key=x, value=x, training=training)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.dense2(self.dropout(self.dense1(x)), training=training)
        x = self.layer_norm2(x + ffn_output)
        return x
    
class Encoder(tfl.Layer):
    def __init__(self,input_dim, d_model, N, heads, dropout):
        super().__init__()
        self.layers = []
        for _ in range(N):
            self.layers.append(EncoderLayer(input_dim, d_model, heads, dropout))
        self.dense = tfl.Dense(3, activation='relu')
    def call(self, x, training=None):
        output = x
        for layer in self.layers:
            output = layer(output, training=training)
        return output
    
class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout):
        super().__init__()
        self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=0.1)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=0.1)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.dense1 = tfl.Dense(d_model, activation='relu')
        self.dense2 = tfl.Dense(output_dim)
        self.layer_norm3 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)
    def call(self, x, encoder_output, training=None):
        attn1 = self.mha1(query=x, key=x, value=x, training=training)
        x = self.layer_norm1(x + self.dropout1(attn1))
        attn2 = self.mha2(query=x, key=encoder_output, value=encoder_output, training=training)
        x = self.layer_norm2(x + self.dropout2(attn2))
        ffn_output = self.dense2(self.dropout3(self.dense1(x)), training=training)
        x = self.layer_norm3(x + ffn_output)
        return x
    
class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.layers = []
        for _ in range(N):
            self.layers.append(DecoderLayer(output_dim, d_model, heads, dropout))
        self.layer_norm = tfl.LayerNormalization()
    def call(self, x, encoder_output, training = None):
        output = x
        for layer in self.layers:
            output = layer(output, encoder_output, training=training)
        output = self.layer_norm(output)
        return output
    
class Transformer(keras.Model):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.activate = tfl.Activation("sigmoid")
    def call(self, x, y, training=None):
        encoder_output = self.encoder(x, training=training)
        decoder_output = self.decoder(y, encoder_output, training=training)
        decoder_output = self.activate(decoder_output)
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

def training_loop(model, train_data_loader, val_data_loader, epochs, loss_fn, optimizer, device, gradient_accumulation_steps=8):
    train_losses = []
    val_losses = []
    with tqdm(total=epochs, unit="epoch") as pbar:
        for epoch in range(epochs):
            # Training phase
            total_train_loss = 0
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
            for step, (src, trg) in enumerate(train_data_loader):
                with tf.device(device):
                    with tf.GradientTape() as tape:
                        output = model(src, trg, training=True)
                        loss = loss_fn(trg, output)
                        loss = loss / gradient_accumulation_steps 

                    # Accumulate gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    for i, (accum_grad, grad) in enumerate(zip(accumulated_gradients, gradients)):
                        if grad is not None:
                            accumulated_gradients[i] += grad

                    if (step + 1) % gradient_accumulation_steps == 0:
                        gradients_and_vars = [
                            (grad, var) for grad, var in zip(accumulated_gradients, model.trainable_variables) if grad is not None
                        ]
                        if gradients_and_vars:
                            optimizer.apply_gradients(gradients_and_vars)
                        accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

                    total_train_loss += loss.numpy()
                    pbar.set_description(f"Train Loss: {total_train_loss / ((step + 1) * gradient_accumulation_steps):.4f}")

            # Validation phase
            model.eval()
            total_val_loss = 0
            for src, trg in val_data_loader:
                # Move the batch to the device
                with tf.device(device):
                    # Forward pass
                    output = model(src, trg, training=False)
                    loss = loss_fn(trg, output)
                    total_val_loss += loss.numpy()
                    pbar.set_description(f"Val Loss: {total_val_loss / len(val_data_loader):.4f}")

            pbar.update(1)
            val_losses.append(total_val_loss / len(val_data_loader))
            train_losses.append(total_train_loss / ((step + 1) * gradient_accumulation_steps))
            print(f"Epoch: {epoch+1} - Train Loss: {total_train_loss/((step + 1) * gradient_accumulation_steps):.4f}, "
                  f"Val Loss: {total_val_loss/len(val_data_loader):.4f}")
        return model, train_losses, val_losses

def evaluate_model(model, test_data_loader, device):
    model.eval()
    true_values = []
    predicted_values = []
    for src, trg in test_data_loader:
        with tf.device(device):
            output = model.call(src, trg)
            true_values.extend(trg.numpy().flatten())
            predicted_values.extend(output.numpy().flatten())
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values)

    return rmse, mae, mape