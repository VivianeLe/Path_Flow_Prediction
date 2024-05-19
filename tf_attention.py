import tensorflow as tf
from keras import layers as tfl
from tqdm.notebook import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Activation

class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_model, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm1 = tfl.LayerNormalization()
        self.batch_norm = BatchNormalization()
        self.activation = Activation('relu')
        self.dense2 = tfl.Dense(input_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dropout = tfl.Dropout(dropout)
        self.layer_norm2 = tfl.LayerNormalization()

    def call(self, x):
        attn_output, attention_scores = self.attn_layer(query=x, key=x, value=x, return_attention_scores=True)
        x = self.layer_norm1(x + self.dropout(attn_output))
        x = self.batch_norm(x)

        ffn_output = self.dense2(self.dropout(self.dense1(x)))
        x = self.layer_norm2(x + ffn_output)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x, attention_scores

class Encoder(tfl.Layer):
    def __init__(self, input_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [EncoderLayer(input_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]
        self.dense = tfl.Dense(3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, x):
        output = x
        attention_scores = []
        for layer in self.layers:
            output, scores = layer(output)
            attention_scores.append(scores)
        return self.dense(output), attention_scores

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout, l2_reg):
        super().__init__()
        self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm1 = tfl.LayerNormalization()
        self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm2 = tfl.LayerNormalization()
        self.batch_norm = BatchNormalization()
        self.dense1 = tfl.Dense(d_model, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        # self.dense2 = tfl.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dense3 = tfl.Dense(output_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.activation = Activation('linear')
        self.layer_norm3 = tfl.LayerNormalization()
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)

    def call(self, x, encoder_output):
        attn1, attn_scores1 = self.mha1(query=x, key=x, value=x, return_attention_scores=True)
        x = self.layer_norm1(x + self.dropout1(attn1))
        x = self.batch_norm(x)

        attn2, attn_scores2 = self.mha2(query=x, key=encoder_output, value=encoder_output, return_attention_scores=True)
        x = self.layer_norm2(x + self.dropout2(attn2))
        x = self.batch_norm(x)
        
        # x2 = self.dropout3(self.dense1(x))
        # x2 = self.dropout3(self.dense2(x2))
        ffn_output = self.dense3(self.dropout3(self.dense1(x)))
        x = self.layer_norm3(x + ffn_output)
        return self.activation(x), [attn_scores1, attn_scores2]

class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout, l2_reg=1e-4):
        super().__init__()
        self.layers = [DecoderLayer(output_dim, d_model, heads, dropout, l2_reg) for _ in range(N)]
        # self.layer_norm = tfl.LayerNormalization()

    def call(self, x, encoder_output):
        output = x
        all_attention_scores = []
        for layer in self.layers:
            output, attention_scores = layer(output, encoder_output)
            all_attention_scores.append(attention_scores)
        # output = self.layer_norm(output)
        return output, all_attention_scores

class Transformer(tfl.Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)

    def call(self, x, y):
        encoder_output, _ = self.encoder(x)
        decoder_output, _ = self.decoder(y, encoder_output)
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
