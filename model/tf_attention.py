import tensorflow as tf
from tensorflow.keras import layers as tfl
from keras import Sequential
from tqdm.notebook import tqdm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Layer, BatchNormalization, Dense, LayerNormalization, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class MultiHeadAttention(Layer):
    def __init__(self, input_dim, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.dense_q = Dense(d_model, use_bias=True)
        self.dense_k = Dense(d_model, use_bias=True)
        self.dense_v = Dense(d_model, use_bias=True)

        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(input_dim, use_bias=True)

    def call(self, queries, keys, values,key_mask, att_mask=True):
        # Linear projections
        Q = self.dense_q(queries)
        K = self.dense_k(keys)
        V = self.dense_v(values)

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
        if att_mask:
            inf = -2**32+1
            key_masks = tf.cast(key_mask, tf.float32)  # (bs, T_k, 1)
            key_masks = tf.transpose(key_masks,[0,2,1]) # (bs, 1, T_k)
            key_masks = tf.tile(key_masks, [self.num_heads, tf.shape(queries)[1],1])  # (bs*h, T_q, T_k)
            paddings = tf.ones_like(outputs) * inf
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Softmax
        outputs = tf.nn.softmax(outputs)

        # Dropout
        outputs = self.dropout(outputs)

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

class EncoderLayer(Layer):
    def __init__(self, input_dim, d_model, heads, dropout, reg_factor):
        super().__init__()
        self.attn_layer = MultiHeadAttention(input_dim, d_model, heads, dropout)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.batch_norm = BatchNormalization()

        self.ffn = Sequential([
            Dense(d_model,  activation='relu', kernel_regularizer=l2(reg_factor)),
            Dense(input_dim),
            Dropout(dropout)
        ])
        self.dropout = Dropout(dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, mask):
        attn_out = self.attn_layer(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_out))

        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x


# class EncoderLayer(tfl.Layer):
#     def __init__(self, input_dim, d_model, heads, dropout, reg_factor):
#         super().__init__()
#         self.dense1 = tf.keras.layers.Dense(d_model, activation='relu', kernel_regularizer=l2(reg_factor))
#         self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
#         self.layer_norm1 = tfl.LayerNormalization()
#         self.dense2 = tfl.Dense(input_dim)
#         self.dropout = tfl.Dropout(dropout)
#         self.layer_norm2 = tfl.LayerNormalization()

#     def call(self, x, training=False):
#         attn_output, _ = self.attn_layer(query=x, key=x, value=x, return_attention_scores=True)
#         x = self.layer_norm1(x + self.dropout(attn_output, training=training))

#         ffn_output = self.dense2(self.dropout(self.dense1(x), training=training))
#         x = self.layer_norm2(x + ffn_output)
#         return x
  
class Encoder(Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout, reg_factor):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout, reg_factor) for _ in range(N)]

    def call(self, x, mask, training=False):
        for layer in self.layers:
            x = layer(x, mask, training=training)
        return x

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout, reg_factor):
        super().__init__()
        self.mha1 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            # Dense(d_model * 2, activation='relu'),
            # Dropout(dropout),
            Dense(d_model, activation='relu',kernel_regularizer=l2(reg_factor)),
            Dense(output_dim),
            Dropout(dropout),
        ])
        self.layer_norm3 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)
        # self.dense = Dense(output_dim, activation='relu')

    def call(self, x, encoder_output, src_mask, tgt_mask):
        # encoder_output = self.dense(encoder_output)
        attn1 = self.mha1(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout1(attn1))

        attn2 = self.mha2(x, encoder_output, encoder_output, src_mask)
        x = self.layer_norm2(x + self.dropout2(attn2))    
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + ffn_output)
        return x

# class DecoderLayer(Layer):
#     def __init__(self, output_dim, d_model, heads, dropout, reg_factor):
#         super().__init__()
#         self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
#         self.layer_norm1 = tfl.LayerNormalization()
#         self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
#         self.layer_norm2 = tfl.LayerNormalization()
#         self.dense1 = tfl.Dense(d_model, activation='relu', kernel_regularizer=l2(reg_factor))
#         self.dense2 = tfl.Dense(output_dim, kernel_regularizer=l2(reg_factor))
#         self.layer_norm3 = tfl.LayerNormalization()
#         self.dropout1 = tfl.Dropout(dropout)
#         self.dropout2 = tfl.Dropout(dropout)
#         self.dropout3 = tfl.Dropout(dropout)

#     def call(self, x, encoder_output, training=False):
#         attn1 = self.mha1(query=x, key=x, value=x)
#         x = self.layer_norm1(x + self.dropout1(attn1, training=training))

#         attn2 = self.mha2(query=x, key=encoder_output, value=encoder_output)
#         x = self.layer_norm2(x + self.dropout2(attn2, training=training))
#         ffn_output = self.dense2(self.dropout3(self.dense1(x), training=training))
#         x = self.layer_norm3(x + ffn_output)
#         return x
    
class Decoder(Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout, reg_factor):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout, reg_factor) for _ in range(N)]
        self.layer_norm = tfl.LayerNormalization()

    def call(self, x, encoder_output, src_mask, tgt_mask, training=False):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, training=training)
        x = self.layer_norm(x)
        return x

class Transformer(Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout, reg_factor):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout, reg_factor)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout, reg_factor)
        self.dense = tfl.Dense(output_dim, activation='linear', kernel_regularizer=l2(reg_factor))

    def call(self, x, y, src_mask, tgt_mask, training=False):
        encoder_output = self.encoder(x, src_mask, training=training)
        decoder_output = self.decoder(y, encoder_output, src_mask, tgt_mask, training=training)
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

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            mode='min',
            min_lr=1e-6
        )
        train_losses = []
        val_losses = []
        with tqdm(total=epochs, unit="epoch") as pbar:
            for epoch in range(epochs):    
                # Training phase
                self.train()
                total_train_loss = 0
                for src, trg, src_mask, tgt_mask in train_data_loader:
                    with tf.device(device):
                        with tf.GradientTape() as tape:
                            output = self.call(src, trg, src_mask, tgt_mask, training=True)
                            loss = loss_fn(trg, output)
                        
                        # Backpropagate and update the model
                        gradients = tape.gradient(loss, self.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                        total_train_loss += loss.numpy()
                        pbar.set_description(f"Train Loss: {total_train_loss / len(train_data_loader):.4f}")
                
                # Validation phase
                self.eval()
                total_val_loss = 0
                for src, trg, src_mask, tgt_mask in val_data_loader:
                    with tf.device(device):
                        output = self.call(src, trg, src_mask, tgt_mask)
                        loss = loss_fn(trg, output)
                        total_val_loss += loss.numpy()
                        
                        pbar.set_description(f"Val Loss: {total_val_loss / len(val_data_loader):.4f}")
                        
                pbar.update(1)
                train_losses.append(total_train_loss / len(train_data_loader))
                val_losses.append(total_val_loss / len(val_data_loader))
                print(f"Epoch: {epoch+1} - Train Loss: {total_train_loss/len(train_data_loader):.4f}, Val Loss: {total_val_loss/len(val_data_loader):.4f}")

                # Check for early stopping
                early_stopping.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                lr_scheduler.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                if early_stopping.stopped_epoch > 0:
                    print(f"Early stopping triggered at epoch {early_stopping.stopped_epoch + 1}")
                    break
        return self, train_losses, val_losses

def evaluate_model(model, test_data_loader, device):
    model.eval()
    total_test_loss = 0
    true_values = []
    predicted_values = []
    for src, trg, src_mask, tgt_mask in test_data_loader:
        with tf.device(device):
            output = model.call(src, trg, src_mask, tgt_mask)
            # loss = loss_fn(trg, output)
            # total_test_loss += loss.numpy()
            true_values.extend(trg.numpy().flatten())
            predicted_values.extend(output.numpy().flatten())

    # test_loss = total_test_loss / len(test_data_loader)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values)

    return rmse, mae, mape