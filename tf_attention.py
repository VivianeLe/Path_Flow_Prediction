import tensorflow as tf
from keras import layers as tfl
from keras import regularizers, Sequential
from tqdm.notebook import tqdm
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Activation, Dense, LayerNormalization, Dropout

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

    def call(self,queries,keys,values,key_mask, att_mask=True):
        # Linear projections,RPE
        Q = self.dense_q(queries) # (N, T_q, d_model)
        K = self.dense_k(keys) # (N, T_k, d_model),learn to rpe
        V = self.dense_v(values) # (N, T_k, d_model),learn to rpe

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
        # print(outputs.shape)
        padding_num = -2 ** 32 + 1 #an inf

        if att_mask:#padding masking
            key_masks = tf.cast(key_mask, tf.float32) # (bs, T_k,1)
            # key_masks = tf.transpose(key_masks,[0,2,1])# (bs, 1,T_k)
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
        outputs = self.dense2(outputs)
        outputs += queries #->(N, T_q, num_units) # do separately in EncoderLayer
        # Normalize
        outputs = self.norm(outputs)
        return outputs
class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.attn_layer = MultiHeadAttention(input_dim, d_model, heads, dropout)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.batch_norm = BatchNormalization()

        self.ffn = Sequential([
            # Dense(d_model * 2, activation='leaky_relu', kernel_regularizer=regularizers.l2(l2_reg)),
            # Dropout(dropout),
            Dense(d_model,  activation='leaky_relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout),
            Dense(input_dim, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.dropout = Dropout(dropout)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        # self.activation = Activation('leaky_relu')

    def call(self, x, mask):
        x = self.attn_layer(x, x, x, mask)
        # x = self.layer_norm1(x + self.dropout(attn_output))
        # x = self.batch_norm(x)

        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        # x = self.activation(x)
        return x

class Encoder(tfl.Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]

    def call(self, x, src_mask):
        output = tf.transpose(x,[0,2,1])
        for layer in self.layers:
            output = layer(output, src_mask)
        output = tf.transpose(output,[0,2,1])
        return output

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.mha1 = MultiHeadAttention(output_dim, d_model, heads, dropout)
        self.layer_norm1 = tfl.LayerNormalization(epsilon=1e-6)
        self.mha2 = MultiHeadAttention(output_dim*3, d_model, heads, dropout)
        self.layer_norm2 = tfl.LayerNormalization(epsilon=1e-6)
        self.ffn = Sequential([
            Dense(d_model * 2, activation='leaky_relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout),
            Dense(d_model,  activation='leaky_relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout),
            Dense(output_dim*3, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        self.layer_norm3 = tfl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)

    def call(self, x, encoder_output, src_mask, tgt_mask):
        x = tf.transpose(x,[0,2,1])
        attn1 = self.mha1(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout1(attn1))
        x = tf.transpose(x,[0,2,1])

        attn2 = self.mha2(x, encoder_output, encoder_output, src_mask, att_mask=False)
        x = self.layer_norm2(x + self.dropout2(attn2))    
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + ffn_output)
        return x
class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]

    def call(self, x, encoder_output, src_mask, tgt_mask):
        # x = tf.transpose(x,[0,2,1])
        output = x
        for layer in self.layers:
            output = layer(output, encoder_output, src_mask, tgt_mask)
        return output

class Transformer(tfl.Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.activation = Activation('linear')

    def call(self, x, y, src_mask, tgt_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(y, encoder_output, src_mask, tgt_mask)
        return self.activation(decoder_output)
    
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
                for src, trg, src_mask, tgt_mask in train_data_loader:
                    with tf.device(device):
                        with tf.GradientTape() as tape:
                            output = self.call(src, trg, src_mask, tgt_mask)
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
                if early_stopping.model is not None:
                    early_stopping.on_epoch_end(epoch, {'val_loss': total_val_loss / len(val_data_loader)})
                    if early_stopping.stopped_epoch > 0:
                        print(f"Early stopping triggered at epoch {early_stopping.stopped_epoch + 1}")
                        break
        return self, train_losses, val_losses