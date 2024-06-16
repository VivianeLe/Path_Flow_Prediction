import tensorflow as tf
from keras import layers as tfl
from keras import regularizers, Sequential
from tqdm.notebook import tqdm
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Activation, Dense, LayerNormalization, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
# from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError, MeanSquaredError
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.mixed_precision import set_global_policy, Policy
policy = Policy('mixed_float16')
set_global_policy(policy)

class MultiHeadAttention(tfl.Layer):
    def __init__(self,input_dim, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.dense_q = Dense(d_model, use_bias=True)
        self.dense_k = Dense(d_model, use_bias=True)
        self.dense_v = Dense(d_model, use_bias=True)
   
        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dense2 = Dense(input_dim, use_bias=True)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)

    def call(self,queries,keys,values,key_mask, att_mask=True, training=None):
        # Linear projections,RPE
        Q = self.dense_q(queries, training=training) # (N, T_q, d_model)
        K = self.dense_k(keys, training=training) # (N, T_k, d_model),learn to rpe
        V = self.dense_v(values, training=training) # (N, T_k, d_model),learn to rpe

        # Split and concat,multi_head,RPE
        Q_= tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h) (256, 625, 64)
        K_= tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        #get attention score->dot_product
        #dot product
        outputs = tf.matmul(Q_,tf.transpose(K_, [0, 2, 1]))# (h*N, T_q, T_k)
        #scale
        d_k = Q_.shape[-1]
        outputs /= d_k ** 0.5
        padding_num = -2 ** 32 + 1 #an inf

        if att_mask:#padding masking
            key_masks = tf.cast(key_mask, tf.float32) # (bs, T_k,1)
            key_masks = tf.transpose(key_masks,[0,2,1])# (bs, 1,T_k)
            key_masks = tf.tile(key_masks, [self.num_heads, tf.shape(queries)[1],1]) # (h*bs, T_q, T_k) (batch size * 8, 625, 625)
            paddings = tf.ones_like(outputs)*padding_num
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
 
        #softmax
        outputs = tf.nn.softmax(outputs)#->(h*batch_size,seq_len,seq_len)

        #dropout
        outputs = self.dropout(outputs)
        #weight_sum->(head*batch,seq_len,num_units/heads)
        outputs = tf.matmul(outputs, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, num_units)
        outputs = self.dense2(outputs, training=training)
        outputs += queries #->(N, T_q, num_units) # do separately in EncoderLayer
        # Normalize
        outputs = self.norm(outputs)
        return outputs
    
class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.attn_layer = MultiHeadAttention(input_dim, d_model, heads, dropout)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.batch_norm = BatchNormalization()

        self.ffn = Sequential([
            Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout),
            Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.dropout = Dropout(dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
    
    def build(self, input_shape):
        self.attn_layer.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, x, mask, training=None):
        x = self.attn_layer(x, x, x, mask, training=training)
        # x = self.layer_norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x, training=training)
        x = self.layer_norm2(x + ffn_output)
        return x

class Encoder(tfl.Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, x, src_mask, training=None):
        for layer in self.layers:
            output = layer(x, src_mask, training=training)
        return output

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.mha1 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            Dense(d_model, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout),
            Dense(output_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.layer_norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def build(self, input_shape):
        self.mha1.build(input_shape)
        self.mha2.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, x, encoder_output, src_mask, tgt_mask, training=None):
        attn1 = self.mha1(x, x, x, tgt_mask, training=training)
        x = self.layer_norm1(x + self.dropout1(attn1))

        attn2 = self.mha2(x, encoder_output, encoder_output, src_mask, att_mask=False, training=training)
        x = self.layer_norm2(x + self.dropout2(attn2))    
        ffn_output = self.ffn(x, training=training)
        x = self.layer_norm3(x + ffn_output)
        return x
    
class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
        super().build(input_shape)

    def call(self, x, encoder_output, src_mask, tgt_mask, training=None):
        output = x
        for layer in self.layers:
            output = layer(output, encoder_output, src_mask, tgt_mask, training=training)
        return output

class Transformer(tfl.Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.activation = sigmoid
    
    def build(self, input_shape):
        encoder_input_shape = input_shape
        decoder_input_shape = (input_shape[0], input_shape[1], self.decoder.layers[0].ffn.layers[-1].units)
        self.encoder.build(encoder_input_shape)
        self.decoder.build(decoder_input_shape)
        super().build(input_shape)

    def call(self, x, y, src_mask, tgt_mask, training=None):
        encoder_output = self.encoder(x, src_mask, training=training)
        decoder_output = self.decoder(y, encoder_output, src_mask, tgt_mask, training=training)
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

    def compile_fit(self, train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device, gradient_accumulation_steps=4):
        input_shape = next(iter(train_data_loader))[0].shape
        self.build(input_shape)

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
                accumulated_gradients = [tf.zeros_like(var) for var in self.trainable_variables]
                # for src, trg, src_mask, tgt_mask in train_data_loader:
                for step, (src, trg, src_mask, tgt_mask) in enumerate(train_data_loader):
                    with tf.device(device):
                        with tf.GradientTape() as tape:
                            output = self.call(src, trg, src_mask, tgt_mask)
                            # tgt_mask = tf.cast(tgt_mask, dtype=output.dtype)
                            loss = loss_fn(trg, output)
                            # loss = tf.reduce_sum(loss) / tf.reduce_sum(tgt_mask)
                            loss = loss / gradient_accumulation_steps
                        
                        # Backpropagate and update the model
                        gradients = tape.gradient(loss, self.trainable_variables)
                        for i, (accum_grad, grad) in enumerate(zip(accumulated_gradients, gradients)):
                            if grad is not None:
                                accumulated_gradients[i] += grad
                        
                        if (step + 1) % gradient_accumulation_steps == 0:
                            gradients_and_vars = [
                                (grad, var) for grad, var in zip(accumulated_gradients, self.trainable_variables) if grad is not None and grad.shape == var.shape
                            ]
                            if gradients_and_vars:
                                optimizer.apply_gradients(gradients_and_vars)
                            accumulated_gradients = [tf.zeros_like(var) for var in self.trainable_variables]

                        total_train_loss += loss.numpy() * gradient_accumulation_steps
                    pbar.set_description(f"Train Loss: {total_train_loss / len(train_data_loader):.4f}")
                
                # Validation phase
                self.eval()
                total_val_loss = 0
                for src, trg, src_mask, tgt_mask in val_data_loader:
                    with tf.device(device):
                        output = self.call(src, trg, src_mask, tgt_mask)
                        # tgt_mask = tf.cast(tgt_mask, dtype=output.dtype)
                        loss = loss_fn(trg, output)
                        # loss = tf.reduce_sum(loss) / tf.reduce_sum(tgt_mask)
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

def evaluate_model(model, test_data_loader, scalers, device, epsilon=1):
    model.eval() 
    true_values = []
    predicted_values = []
    scaler_idx = 0
    for batch in test_data_loader:
        src, trg, src_mask, tgt_mask = batch
        with tf.device(device):
            output = model(src, trg, src_mask, tgt_mask, training=False)
            tgt_mask = tf.cast(tgt_mask, dtype=output.dtype)
            output = output * tgt_mask

            # Convert predictions and true values back to original scale using the scalers
            for i in range(len(src)):
                scaler = scalers[scaler_idx]
                scaler_idx += 1
                trg_original = scaler.inverse_transform(trg[i].numpy().reshape(-1, trg.shape[-1]))
                output_original = scaler.inverse_transform(output[i].numpy().reshape(-1, output.shape[-1]))

                true_values.extend(trg_original)
                predicted_values.extend(output_original)

    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    mape = np.mean(np.abs((np.array(true_values) - np.array(predicted_values)) / (np.array(true_values) + epsilon))) * 100

    return mse, rmse, mae, mape

# def evaluate_model(model, test_data_loader, device):
#     model.eval()
#     true_values = []
#     predicted_values = []
#     for src, trg, src_mask, tgt_mask, in test_data_loader:
#         with tf.device(device):
#             output = model.call(src, trg, src_mask, tgt_mask, training=False)
#             true_values.extend(trg.numpy().flatten())
#             predicted_values.extend(output.numpy().flatten())
#     mse = mean_squared_error(true_values, predicted_values)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(true_values, predicted_values)

#     return mse, rmse, mae
    
# def training_loop(model, train_data_loader, val_data_loader, optimizer, loss_fn, epochs, device, gradient_accumulation_steps=8):
#     train_losses = []
#     val_losses = []
#     with tqdm(total=epochs, unit="epoch") as pbar:
#         for epoch in range(epochs):
#             # Training phase
#             total_train_loss = 0
#             accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
#             for step, (src, trg, src_mask, tgt_mask) in enumerate(train_data_loader):
#                 with tf.device(device):
#                     with tf.GradientTape() as tape:
#                         output = model(src, trg, src_mask, tgt_mask, training=True)
#                         loss = loss_fn(trg, output)
#                         loss = loss / gradient_accumulation_steps 

#                     # Accumulate gradients
#                     gradients = tape.gradient(loss, model.trainable_variables)
#                     for i, (accum_grad, grad) in enumerate(zip(accumulated_gradients, gradients)):
#                         if grad is not None:
#                             accumulated_gradients[i] += grad

#                     if (step + 1) % gradient_accumulation_steps == 0:
#                         gradients_and_vars = [
#                             (grad, var) for grad, var in zip(accumulated_gradients, model.trainable_variables) if grad is not None
#                         ]
#                         if gradients_and_vars:
#                             optimizer.apply_gradients(gradients_and_vars)
#                         accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

#                     total_train_loss += loss.numpy()
#                     pbar.set_description(f"Train Loss: {total_train_loss / ((step + 1) * gradient_accumulation_steps):.4f}")

#             # Validation phase
#             model.eval()
#             total_val_loss = 0
#             for src, trg, src_mask, tgt_mask in val_data_loader:
#                 # Move the batch to the device
#                 with tf.device(device):
#                     # Forward pass
#                     output = model(src, trg, src_mask, tgt_mask, training=False)
#                     loss = loss_fn(trg, output)
#                     total_val_loss += loss.numpy()
#                     pbar.set_description(f"Val Loss: {total_val_loss / len(val_data_loader):.4f}")

#             pbar.update(1)
#             val_losses.append(total_val_loss / len(val_data_loader))
#             train_losses.append(total_train_loss / ((step + 1) * gradient_accumulation_steps))
#             print(f"Epoch: {epoch+1} - Train Loss: {total_train_loss/((step + 1) * gradient_accumulation_steps):.4f}, "
#                   f"Val Loss: {total_val_loss/len(val_data_loader):.4f}")
#         return model, train_losses, val_losses
