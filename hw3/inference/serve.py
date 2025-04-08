import tensorflow as tf
import numpy as np
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)

# Global variable for model
model = None

def load_model():
    global model
    model_path = '/models/mnist_model_latest.keras'
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

@app.route('/')
def home():
    return '''
    <h1>MNIST Digit Classification Service</h1>
    <p>API endpoints:</p>
    <ul>
        <li>/predict/digit - Send a POST request with an image file</li>
        <li>/predict/random - Send a GET request to classify a random test digit</li>
        <li>/status - Check if the service is running</li>
    </ul>
    <form action="/predict/digit" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload and predict">
    </form>
    '''

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None
    })

@app.route('/predict/random')
def predict_random():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Load a small sample from MNIST test set
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255.0

    # Choose a random image
    index = np.random.randint(0, len(x_test))
    image = x_test[index]
    true_label = int(y_test[index])

    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = int(np.argmax(prediction))

    return jsonify({
        'predicted_digit': predicted_label,
        'true_digit': true_label,
        'confidence': float(tf.nn.softmax(prediction)[0][predicted_label])
    })

@app.route('/predict/digit', methods=['POST'])
def predict_digit():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        img = Image.open(file.stream).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST format
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        predicted_label = int(np.argmax(prediction))
        confidence = float(tf.nn.softmax(prediction)[0][predicted_label])

        return jsonify({
            'predicted_digit': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001)