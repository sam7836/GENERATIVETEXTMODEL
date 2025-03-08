import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import random

# 🔹 Sample Training Data (You can replace with a larger dataset)
corpus = [
    "Artificial Intelligence is transforming industries",
    "Deep learning is a subset of machine learning",
    "Machine learning powers many AI applications",
    "Neural networks are used in deep learning",
    "AI is enabling automation across the world",
    "Natural Language Processing helps computers understand human language",
    "Computer vision enables machines to see and interpret images",
    "Reinforcement learning allows AI to learn from experience",
    "AI ethics is an important field to ensure responsible AI usage",
    "Big data and AI together provide valuable business insights",
    "Self-driving cars use deep learning for real-time decision making",
    "AI chatbots improve customer support efficiency",
    "Speech recognition converts spoken words into text",
    "Generative AI models can create realistic images and text",
    "AI helps in medical diagnosis by analyzing patient data",
    "Predictive analytics powered by AI helps in decision making",
]


# 🔹 Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# 🔹 Convert Text to Sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# 🔹 Pad Sequences to Equal Length
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# 🔹 Split Data into Features (xs) and Labels (ys)
xs, ys = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(ys, num_classes=total_words)

# 🔹 Define the LSTM Model
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    LSTM(256, return_sequences=True, dropout=0.2),  # Dropout to prevent overfitting
    LSTM(256, dropout=0.2),
    Dense(256, activation='relu'),
    Dense(total_words, activation='softmax')  # Output layer with softmax activation
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔹 Train the Model
print("Training LSTM model...")
model.fit(xs, ys, epochs=200, verbose=1)  # Increase epochs for better results

# 🔹 Function to Generate Text using Temperature Sampling
def generate_text(seed_text, next_words=10, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.asarray(predictions).astype("float64")

        # Apply temperature scaling
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predictions), p=predictions)
        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word
    return seed_text

# 🔹 Test the Model with Text Generation
print("\nGenerated Text: ", generate_text("Artificial Intelligence", next_words=10, temperature=1.0))
