import tensorflow as tf
import os
import time

print("TensorFlow version:", tf.__version__)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train the model
print("Starting model training...")
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
print("Evaluating model performance...")
model.evaluate(x_test, y_test, verbose=2)

# Save the model to shared volume
models_dir = '/models'
os.makedirs(models_dir, exist_ok=True)

timestamp = int(time.time())
model_path = os.path.join(models_dir, f'mnist_model_latest.keras')
print(f"Saving model to {model_path}")
print(f"model_path actually is: {model_path}")  # Ensure it includes `.keras`

model.save(model_path)

# Also save a symlink to the latest model
latest_path = os.path.join(models_dir, 'mnist_model_latest')
if os.path.exists(latest_path) or os.path.islink(latest_path):
    os.remove(latest_path)
os.symlink(model_path, latest_path)

print("Training completed successfully!")