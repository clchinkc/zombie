import builtins
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import LambdaCallback, TensorBoard
from keras.datasets import mnist
from keras.layers import BatchNormalization, Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from tensorboard import program
from webdriver_manager.chrome import ChromeDriverManager


# Function to plot images and convert to tensor
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue())
    image = tf.expand_dims(image, 0)
    return image

# Function to create an image grid
def image_grid(epoch):
    figure = plt.figure(figsize=(10, 4))
    n = 10  # number of digits to display
    decoded_imgs = autoencoder.predict(x_test[:n], verbose=0)
    for i in range(n):
        # Original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(f'Epoch {epoch}')
    return figure

def run_tensorboard(logdir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    tb.launch()
    tensorboard_url = 'http://localhost:6006/'
    print(f"TensorBoard is running at {tensorboard_url}")

    # Start Chrome with Selenium
    chrome_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=chrome_service)
    driver.get(tensorboard_url)
    print(f"Chrome started at {tensorboard_url}")

    return driver

if __name__ == '__main__':
    # Load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    # Autoencoder Model
    input_img = Input(shape=(28, 28, 1))
    x = BatchNormalization()(input_img)
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    autoencoder.summary()

    # Set up TensorBoard
    log_dir = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
    file_writer_image = tf.summary.create_file_writer(log_dir + '/image')

    def log_image(epoch, logs):
        # Log the image grid as an image summary.
        figure = image_grid(epoch)
        image = plot_to_image(figure)

        with file_writer_image.as_default(step=epoch):
            tf.summary.image("Original vs Reconstructed", image, step=epoch)

    image_callback = LambdaCallback(on_epoch_end=log_image)

    # Train the model
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=1024,
                    verbose=0,
                    validation_data=(x_test, x_test),
                    callbacks=[tensorboard_callback, image_callback])

    # Evaluate the model
    score = autoencoder.evaluate(x_test, x_test, verbose=0)
    print('Test loss:', score)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_tensorboard, log_dir)
        driver = future.result()

        builtins.input("Press Enter to stop TensorBoard and close the Chrome window.")

        # Close Chrome
        driver.quit()


