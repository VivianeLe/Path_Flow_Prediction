import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from keras import layers as tfl

class EncoderLayer(tfl.Layer):
    def __init__(self, input_dim, d_model, heads, dropout):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(d_model, activation='relu')
        self.attn_layer = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=2)
        self.layer_norm1 = tfl.LayerNormalization()
        self.dense2 = tfl.Dense(input_dim)
        self.dropout = tfl.Dropout(dropout)
        self.layer_norm2 = tfl.LayerNormalization()

    def call(self, x, training=None):
        # Assuming x has a shape of (batch_size, sequence_length, input_dim)
        attn_output, attention_scores = self.attn_layer(query=x, key=x, value=x, return_attention_scores=True, training=training)
        x = self.layer_norm1(x + self.dropout(attn_output), training=training)

        ffn_output = self.dense2(self.dropout(self.dense1(x)), training=training)
        x = self.layer_norm2(x + ffn_output)
        
        return x, attention_scores
    
class Encoder(tfl.Layer):
    def __init__(self,input_dim, d_model, N, heads, dropout):
        super().__init__()
        self.layers = []
        for _ in range(N):
            self.layers.append(EncoderLayer(input_dim, d_model, heads, dropout))
        self.dense = tfl.Dense(3, activation='relu')

    def call(self, x, training=None):
        output = x
        attention_scores = []
        for layer in self.layers:
            output, scores = layer(output, training=training)
            attention_scores.append(scores)
        return self.dense(output), attention_scores

class DecoderLayer(tfl.Layer):
    def __init__(self, output_dim, d_model, heads, dropout):
        super().__init__()
        self.mha1 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
        self.layer_norm1 = tfl.LayerNormalization()
        self.mha2 = tfl.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, attention_axes=(1,))
        self.layer_norm2 = tfl.LayerNormalization()
        self.dense1 = tfl.Dense(d_model, activation='relu')
        self.dense2 = tfl.Dense(output_dim)
        self.layer_norm3 = tfl.LayerNormalization()
        self.dropout1 = tfl.Dropout(dropout)
        self.dropout2 = tfl.Dropout(dropout)
        self.dropout3 = tfl.Dropout(dropout)

    def call(self, x, encoder_output, training=None):
        attn1, attn_scores1 = self.mha1(query=x, key=x, value=x, return_attention_scores=True, training=training)
        x = self.layer_norm1(x + self.dropout1(attn1), training=training)

        # Encoder-decoder attention
        attn2, attn_scores2 = self.mha2(query=x, key=encoder_output, value=encoder_output, return_attention_scores=True, training=training)
        x = self.layer_norm2(x + self.dropout2(attn2), training=training)
        ffn_output = self.dense2(self.dropout3(self.dense1(x)), training=training)
        x = self.layer_norm3(x + ffn_output)
        return x, [attn_scores1, attn_scores2]

class Decoder(tfl.Layer):
    def __init__(self, output_dim, d_model, N, heads, dropout):
        super().__init__()
        
        self.layers = []
        for _ in range(N):
            self.layers.append(DecoderLayer(output_dim, d_model, heads, dropout))
        self.layer_norm = tfl.LayerNormalization()

    def call(self, x, encoder_output, training = None):
        output = x
        all_attention_scores = []
        for layer in self.layers:
            output, attention_scores = layer(output, encoder_output, training=training)
            all_attention_scores.append(attention_scores)
        output = self.layer_norm(output)
        return output, all_attention_scores
    
class Transformer(tfl.Layer):
    def __init__(self, input_dim, output_dim, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(output_dim, d_model, N, heads, dropout)
        self.dense = tfl.Dense(output_dim, activation='linear')
        self.is_training = True

    def call(self, x, y, training=None):
        encoder_output, _ = self.encoder(x, training=training)
        decoder_output, _ = self.decoder(y, encoder_output, training=training)
        return self.dense(decoder_output)

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
