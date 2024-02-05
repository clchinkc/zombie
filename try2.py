import builtins
import io
from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard import default, program
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Function to define custom metric
def custom_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Function to plot images and convert to tensor
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

# Function to create an image grid
def image_grid():
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    figure = plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.binary)
    return figure

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create a Sequential model
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', custom_metric])

# Set up TensorBoard
log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)

# Train the model
model.fit(x_train, y_train, 
          epochs=3, 
          batch_size=64,
          steps_per_epoch=50,
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard])

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

file_writer = tf.summary.create_file_writer(log_dir)
figure = image_grid()
with file_writer.as_default():
    tf.summary.image("Training data", plot_to_image(figure), step=0)

tb = program.TensorBoard(plugins=default.get_plugins())
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard is running at {url}")
# Wait for user to close TensorBoard
print("Press Enter to stop TensorBoard and exit the script.")
builtins.input()
# After pressing Enter, the script will continue from here
print("Stopping TensorBoard and exiting the script.")
