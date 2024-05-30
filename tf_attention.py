import pandas as pd
import tensorflow as tf
from keras import layers as tfl
from keras import regularizers, Sequential
from tqdm.notebook import tqdm
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Activation, Dense, LayerNormalization, Dropout

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.dense_q = Dense(d_model, use_bias=True)
        self.dense_k = Dense(d_model, use_bias=True)
        self.dense_v = Dense(d_model, use_bias=True)

        # Create learnable tensor parameters for K, Q, and V
        # self.W_q = self.add_weight(shape=(input_dim, d_model),
        #                           initializer="glorot_uniform",
        #                           trainable=True,
        #                           name="W_q")
        # self.W_k = self.add_weight(shape=(input_dim, d_model),
        #                           initializer="glorot_uniform",
        #                           trainable=True,
        #                           name="W_k")
        # self.W_v = self.add_weight(shape=(input_dim, d_model),
        #                           initializer="glorot_uniform",
        #                           trainable=True,
        #                           name="W_v")

        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(input_dim, use_bias=True)

    def call(self, queries, keys, values):
        # Linear projections
        Q = self.dense_q(queries)
        K = self.dense_k(keys)
        V = self.dense_v(values)

        # Q = tf.matmul(queries, self.W_q)  # (N, T_q, d_model)
        # K = tf.matmul(keys, self.W_k)  # (N, T_k, d_model)
        # V = tf.matmul(values, self.W_v)  # (N, T_k, d_model)

        # Split and concat, multi_head
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention score, dot product
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        # Scale
        d_k = Q_.shape[-1]
        outputs /= d_k ** 0.5

        # Padding masking
        # if att_mask:
        #     key_masks = tf.cast(key_mask, tf.float32)  # (bs, 1, T_k)
        #     key_masks = tf.tile(key_masks, [self.num_heads, tf.shape(queries)[1], 1])  # (bs*h, T_q, T_k)
        #     paddings = tf.ones_like(outputs) * -2 ** 32 + 1
        #     outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Softmax
        outputs = tf.nn.softmax(outputs)

        # Dropout
        # outputs = self.dropout(outputs)

        # Weight sum
        outputs = tf.matmul(outputs, V_) # (h*N, T_k, d_model/h) 

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, num_units)
        outputs = self.dense2(outputs)

        # Residual block
        # outputs += queries  # (N, T_q, num_units)

        # Normalize
        # outputs = self.norm(outputs)
        return outputs

# class EncoderLayer(tfl.Layer):
#     def __init__(self, input_dim, d_model, heads, dropout, l2_reg):
#         super().__init__()
#         self.dense1 = tf.keras.layers.Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))
#         self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
#         self.layer_norm1 = tfl.LayerNormalization()
#         self.dense2 = tfl.Dense(input_dim)
#         self.dropout = tfl.Dropout(dropout)
#         self.layer_norm2 = tfl.LayerNormalization()

#     def call(self, x):
#         # Assuming x has a shape of (batch_size, sequence_length, input_dim)
#         attn_output, _ = self.attn_layer(query=x, key=x, value=x, return_attention_scores=True)
#         x = self.layer_norm1(x + self.dropout(attn_output))

#         ffn_output = self.dense2(self.dropout(self.dense1(x)))
#         x = self.layer_norm2(x + ffn_output)
        
#         return x
    
class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout):
        super().__init__()
        self.attn_layer = MultiHeadAttention(input_dim, d_model, heads, dropout)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.batch_norm = BatchNormalization()

        self.ffn = Sequential([
            Dense(d_model,  activation='relu'),
            Dense(input_dim),
            Dropout(dropout)
        ])
        self.dropout = Dropout(dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_out = self.attn_layer(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))

        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class Encoder(tfl.Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout) for _ in range(N)]
        # self.dense = tfl.Dense(3)

    def call(self, x):
        output=x
        # output = tf.transpose(x,[0,2,1])
        for layer in self.layers:
            output = layer(output)
        # output = tf.transpose(output,[0,2,1])
        return output

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            # Dense(d_model * 2, activation='relu'),
            # Dropout(dropout),
            Dense(d_model, activation='relu'),
            Dense(output_dim),
            Dropout(dropout),
        ])
        self.layer_norm3 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)
        self.dense = Dense(output_dim, activation='relu')

    def call(self, x, encoder_output):
        # encoder_output = self.dense(encoder_output)
        attn1 = self.mha1(x, x, x)
        x = self.layer_norm1(x + self.dropout1(attn1))

        attn2 = self.mha2(x, encoder_output, encoder_output)
        x = self.layer_norm2(x + self.dropout2(attn2))    
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + ffn_output)
        return x

# class DecoderLayer(tfl.Layer):
#     def __init__(self, output_dim, d_model, heads, dropout, l2_reg):
#         super().__init__()
#         self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
#         self.layer_norm1 = tfl.LayerNormalization()
#         self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
#         self.layer_norm2 = tfl.LayerNormalization()
#         self.dense1 = tfl.Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))
#         self.dense2 = tfl.Dense(output_dim)
#         self.layer_norm3 = tfl.LayerNormalization()
#         self.dropout1 = tfl.Dropout(dropout)
#         self.dropout2 = tfl.Dropout(dropout)
#         self.dropout3 = tfl.Dropout(dropout)

#     def call(self, x, encoder_output):
#         attn1, _ = self.mha1(query=x, key=x, value=x, return_attention_scores=True)
#         x = self.layer_norm1(x + self.dropout1(attn1))

#         # Encoder-decoder attention
#         attn2, _ = self.mha2(query=x, key=encoder_output, value=encoder_output, return_attention_scores=True)
#         x = self.layer_norm2(x + self.dropout2(attn2))
#         ffn_output = self.dense2(self.dropout3(self.dense1(x)))
#         x = self.layer_norm3(x + ffn_output)
#         return x
    
class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout) for _ in range(N)]
        self.layer_norm = tfl.LayerNormalization()

    def call(self, x, encoder_output, training=None):
        output = x
        for layer in self.layers:
            output = layer(output, encoder_output, training=training)
        output = self.layer_norm(output)
        return output

class Transformer(tfl.Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.dense = tfl.Dense(output_dim, activation='linear')
        self.is_training = True

    def call(self, x, y):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        output = self.dense(decoder_output)
        return output
    
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
    
    def compile(self, train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device):
        # Define the early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        train_losses = []
        val_losses = []
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
                        pbar.set_description(f"Train Loss: {total_train_loss / len(train_data_loader):.4f}")
                
                # Validation phase
                self.eval()
                total_val_loss = 0
                for src, trg in val_data_loader:
                    with tf.device(device):
                        output = self.call(src, trg)
                        loss = loss_fn(trg, output)
                        total_val_loss += loss.numpy()
                        
                        pbar.set_description(f"Val Loss: {total_val_loss / len(val_data_loader):.4f}")
                        
                pbar.update(1)
                train_losses.append(total_train_loss / len(train_data_loader))
                val_losses.append(total_val_loss / len(val_data_loader))
                print(f"Epoch: {epoch+1} - Train Loss: {total_train_loss/len(train_data_loader):.4f}, Val Loss: {total_val_loss/len(val_data_loader):.4f}")

                # Check for early stopping
                if early_stopping.model is not None:
                    early_stopping.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                    if early_stopping.stopped_epoch > 0:
                        print(f"Early stopping triggered at epoch {early_stopping.stopped_epoch + 1}")
                        break
        return self, train_losses, val_losses