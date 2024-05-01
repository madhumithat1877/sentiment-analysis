import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
#from sentiment_utils import *
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

#from emo_utils import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.layers import  Conv1D, MaxPooling1D, Bidirectional, Flatten, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Constant
import nltk
nltk.download('stopwords')

# Load the training and test data from CSV files
train_df = pd.read_csv('tamil_movie_reviews_train.csv')
test_df = pd.read_csv('tamil_movie_reviews_test.csv')

def preprocess_data(df):
    # removal of newline
    nl = "<NEWLINE>"

    # Function for removing punctuation
    def newline_remove(text_data):
        # Appending non punctuated words
        newline ="".join([t for t in text_data if t not in nl])
        return newline

    # Passing input to the function
    df["newline_removed"] = df['ReviewInTamil'].apply(newline_remove)


    #remove 'avs' from strings in team column
    df['newline_removed'] = df['newline_removed'].str.replace('Review in nglish', '')

    #remove 'avs' from strings in team column
    df['newline_removed'] = df['newline_removed'].str.replace('Read', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Karuppan', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Velaikkaran', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Baahubali 2', '')
    df['newline_removed'] = df['newline_removed'].str.replace('AAA', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Kaatru Veliyidai', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Adhe Kangal', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Mo', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Peechankai vie', '')
    df['newline_removed'] = df['newline_removed'].str.replace('Vanamagan', '')
    df['newline_removed'] = df['newline_removed'].str.replace('spyder tamil movie review rating live updates', '')
    df['newline_removed'] = df['newline_removed'].str.replace('quotnkitta thatha vie review', '')
    df['newline_removed'] = df['newline_removed'].str.replace('quot', '')

    # Importing python string function
    import string
    # Printing Inbuilt punctuation function
    print(string.punctuation)

    # Function for removing punctuation
    def punctuation_remove(text_data):
        # Appending non punctuated words
        punctuation =''.join([t for t in text_data if t not in string.punctuation])
        return punctuation

    # Passing input to the function
    df["punct_removed"] = df['newline_removed'].apply(punctuation_remove)
    df = df.drop(['ReviewInTamil'], axis=1)
    df = df.drop(['newline_removed'], axis=1)


    # Remove standard spaces
    df['punct_removed'] = df['punct_removed'].str.strip()

    # Remove tabs and new lines as well
    df['punct_removed'] = df['punct_removed'].str.replace(r'\t|\n|\r', '', regex=True)


    # Assuming 'punct_removed' is the column containing the processed text
    df['punct_removed'] = df['punct_removed'].astype(str)

    # # Download Tamil stopwords
    # stop_words_tamil = set(stopwords.words('tamil'))

    # # Function to remove stopwords
    # def remove_stopwords(text):
    #     words = word_tokenize(text)
    #     filtered_words = [word for word in words if word.lower() not in stop_words_tamil]
    #     return ' '.join(filtered_words)

    # # Apply the function to the 'punct_removed' column
    # df['punct_removed'] = df['punct_removed'].apply(remove_stopwords)

    # Manually define Tamil stopwords
    stop_words_tamil = set([
        'அந்த', 'அவன்', 'அவனது', 'அவன்னை', 'ஒரு','என்று','மற்றும்','இந்த','இது','என்ற','கொண்டு','என்பது',
        'பல','ஆகும்','அல்லது','அவர்','நான்','உள்ள','அந்த','இவர்','என','முதல்','என்ன','இருந்து','சில','என்',
        'போன்ற', 'வேண்டும்','வந்து','இதன்','அது','அவன்','தான்','பலரும்','என்னும்','மேலும்','பின்னர்',
        'கொண்ட','இருக்கும்','தனது','உள்ளது','போது','என்றும்','அதன்','தன்','பிறகு','அவர்கள்','வரை','அவள்',
        'நீ','ஆகிய','இருந்தது','உள்ளன', 'வந்த','இருந்த','மிகவும்','இங்கு','மீது','ஓர்','இவை','இந்தக்','பற்றி',
        'வரும்','வேறு','இரு','இதில்','போல்','இப்போது','அவரது','மட்டும்','இந்தப்','எனும்','மேல்','பின்','சேர்ந்த','ஆகியோர்','எனக்கு','இன்னும்','அந்தப்','அன்று',
        'ஒரே','மிக','அங்கு','பல்வேறு','விட்டு','பெரும்','அதை','பற்றிய','அதிக','அந்தக்','பேர்','இதனால்',
        'அவை','அதே','ஏன்', 'முறை','யார்','என்பதை','எல்லாம்','மட்டுமே','இங்கே','அங்கே','இடம்','இடத்தில்',
        'அதில்','நாம்','அதற்கு', 'எனவே','பிற','சிறு','மற்ற','விட','எந்த','எனவும்','எனப்படும்','எனினும்',
        'அடுத்த','இதனை','இதை', 'கொள்ள','இந்தத்','இதற்கு','அதனால்','தவிர','போல','வரையில்','சற்று',
        'எனக்','கரு'])

    # Function to remove stopwords
    def remove_tamil_stopwords(text):
        words = [word for word in text.split() if word.lower() not in stop_words_tamil]
        return ' '.join(words)

    # Apply the function to your 'punct_removed' column
    df['punct_removed'] = df['punct_removed'].apply(remove_tamil_stopwords)


    #cleaning Data
    df = df[['punct_removed', 'Rating']]
    df.loc[:,'new_rating'] = df['Rating'].apply(lambda x: 0 if x<=3 else 1)
    df = df.drop(['Rating'], axis=1)
    # print(df)
    return df

# Now you can use the text_data with TensorFlow for further processing
# For example, tokenization or any other NLP task
###################################################################################
# Preprocess the training data
train_df = preprocess_data(train_df)

# Preprocess the test data
test_df = preprocess_data(test_df)

# Initialize the Tokenizer
tokenizer = Tokenizer()

# Fit tokenizer on training data
tokenizer.fit_on_texts(train_df['punct_removed'])

# Convert text data to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['punct_removed'])
test_sequences = tokenizer.texts_to_sequences(test_df['punct_removed'])

# Pad sequences
max_len = max(len(seq) for seq in train_sequences + test_sequences)
print("Maximum sequence length:", max_len)
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

# Split the data into features and labels
X_train, y_train = train_padded_sequences, train_df['new_rating']
X_test, y_test = test_padded_sequences, test_df['new_rating']

# ######################################################################################
# Create vocabulary
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print("Vocab Size", vocab_size)
# Download the GloVe word embeddings

# Train a GloVe model
# You can use the embedding_matrix as initial weights in your model
# Then compile and train your model as usual

# This code snippet performs tokenization, padding, and creates a vocabulary for your text data. It also loads the GloVe word embeddings,
# creates an embedding matrix, and prepares it for training your model. Finally, you can use this embedding matrix as initial weights in your model
# and train it as usual. Adjust the parameters like max_len, embedding_dim, and file paths according to your specific requirements.


# Load GloVe embeddings into memory
embedding_dim = 200  # Change according to the dimension of your GloVe embeddings
embeddings_index = {}
batch_size=32
glove_file = "glove.6B.200d.txt"
with open(glove_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Save the embedding matrix
np.save('embedding_matrix.npy', embedding_matrix)

# Define the model architecture: CNN-BiLSTM
embedding_dim = 200
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,embeddings_initializer=Constant(embedding_matrix), trainable=False))
model.add(SpatialDropout1D(0.1))
model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Activation('relu'))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(96)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# Build the model to finalize its architecture
model.build(input_shape=(batch_size, max_len))
# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Print model summary
print(model.summary())

from sklearn.metrics import classification_report, confusion_matrix

# Predict probabilities for test set
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = (y_pred_prob > 0.5).astype(int)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Plotting the loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from tensorflow.keras.models import load_model

# Assuming 'model' is your trained CNN-BiLSTM model
model.save('cnn_bilstm_model.h5')
