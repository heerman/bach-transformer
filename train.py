# A Bach transformer model
import music21
import tensorflow as tf

# Training data as a sequence of MIDI numbers
midi_sequence = []
sequence_length = 5

# Bach corpus from music21
bwv846 = music21.corpus.parse('bach/bwv846')
for part in bwv846.parts:
    for measure in part.measures(1, None):
        if isinstance(measure, music21.stream.base.Measure):
            for element in measure.elements:
                if isinstance(element, music21.note.Note):
                    midi_sequence.append(element.nameWithOctave)

print("Training data: ", midi_sequence)

# Prepare the sequences as input for the model
sample_input = []
for i in range(len(midi_sequence) - sequence_length):
    # Training data (shape = [batch_size, sequence_length])
    seq_in = midi_sequence[i:i + sequence_length]
    sample_input.append(seq_in)

# Converting to TensorFlow tensor
sample_input = tf.constant(sample_input)

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
