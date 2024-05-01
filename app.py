from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from  process import process
app = Flask(__name__)

# Load the trained model
model = load_model('cnn_bilstm_model.h5',compile=False)

# Initialize the Tokenizer
tokenizer = Tokenizer()

# Define the maximum sequence length
max_len = 928  #model's input shape

# Load the embedding matrix
embedding_dim = 200  #  GloVe embeddings dimension
vocab_size = 40681  # tokenizer's vocabulary size

# Load the embedding matrix
embedding_matrix = np.load('embedding_matrix.npy')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        text = request.form['text'].strip()

        # Preprocess the input text
        text_sequence = tokenizer.texts_to_sequences([text])
        text_sequence_padded = pad_sequences(text_sequence, maxlen=max_len, padding='post')

        # Predict the sentiment
        prediction = model.predict(text_sequence_padded)
        prediction=process(text)
        # Map prediction to sentiment label
        print(prediction)
        sentiment = "Positive" if prediction  else "Negative"

        return render_template('result.html', sentiment=sentiment )

if __name__ == '__main__':
    app.run(debug=True)
