# Training a transformer model on Bach
import tensorflow as tf

# Sample data (shape = [batch_size, sequence_length])
sample_input = tf.constant([[1, 2, 3], [4, 5, 6]])

# Parameters
embedding_dim = 64
num_heads = 2
ff_dim = 32

# Define the input layer
input_layer = tf.keras.Input(shape=(None,))

# Embedding layer
embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=embedding_dim)
x = embedding_layer(input_layer)

# Multi-Head Attention
attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
attention_output = attention(x, x)

# Feed Forward Network
ffn = tf.keras.Sequential([
    tf.keras.layers.Dense(ff_dim, activation="relu"),
    tf.keras.layers.Dense(embedding_dim),
])
ffn_output = ffn(attention_output)

# Final Model
model = tf.keras.Model(inputs=input_layer, outputs=ffn_output)
model.summary()
