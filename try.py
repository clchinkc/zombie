import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
    Softmax,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


# Define the generator
def build_generator(latent_dim, data_shape):
    model = Sequential()
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(128, activation="elu"))
    model.add(BatchNormalization())
    model.add(Dense(np.prod(data_shape) * 4))  # Multiply by 4 for one-hot encoding
    model.add(Reshape((*data_shape, 4)))
    model.add(Softmax(axis=-1))
    return model

# Define the critic
def build_critic(data_shape):
    model = Sequential()
    model.add(Input(shape=(*data_shape, 4)))
    # Apply weight clipping in Conv2D layers
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation="elu", kernel_constraint=lambda w: tf.clip_by_value(w, -0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation="elu", kernel_constraint=lambda w: tf.clip_by_value(w, -0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))  # Linear activation
    return model

# Update GAN building function
def build_wgan(generator, critic):
    # Wasserstein loss function
    def wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    critic.compile(loss=wasserstein_loss, optimizer=RMSprop())
    critic.trainable = False

    gan_input = Input(shape=(latent_dim,))
    gan_output = critic(generator(gan_input))

    gan = Model(gan_input, gan_output)
    gan.compile(loss=wasserstein_loss, optimizer=RMSprop())

    return gan

# Update training procedure
def train_wgan(gan, generator, critic, latent_dim, epochs, batch_size, data_shape, critic_interval=2, generator_interval=1):
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))

    for epoch in range(epochs):
        c_loss_real = 0
        c_loss_fake = 0
        g_loss_total = 0
        
        for _ in range(critic_interval):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise, verbose=0)
            real_imgs = to_categorical(np.random.randint(0, 4, (batch_size, *data_shape)), num_classes=4)

            real_dataset = tf.data.Dataset.from_tensor_slices((real_imgs, valid)).batch(batch_size)
            fake_dataset = tf.data.Dataset.from_tensor_slices((gen_imgs, fake)).batch(batch_size)

            critic.trainable = True
            c_real_loss = critic.fit(real_dataset, epochs=critic_interval, verbose=0)
            c_fake_loss = critic.fit(fake_dataset, epochs=critic_interval, verbose=0)
            c_loss_real += c_real_loss.history['loss'][-1]
            c_loss_fake += c_fake_loss.history['loss'][-1]


        for _ in range(generator_interval):
            critic.trainable = False
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            noise_dataset = tf.data.Dataset.from_tensor_slices((noise, valid)).batch(batch_size)
            g_loss = gan.fit(noise_dataset, epochs=generator_interval, verbose=0)
            g_loss_total += g_loss.history['loss'][-1]

        c_loss_real_avg = c_loss_real / critic_interval
        c_loss_fake_avg = c_loss_fake / critic_interval
        g_loss_avg = g_loss_total / generator_interval

        print(f"Epoch {epoch}/{epochs} [Critic: real loss: {c_loss_real_avg:.4f}, fake loss: {c_loss_fake_avg:.4f}] [Generator loss: {g_loss_avg:.4f}]")

    sample_noise = np.random.normal(0, 1, (1, latent_dim))
    generated_data = generator.predict(sample_noise, verbose=0)
    generated_data_class = np.argmax(generated_data, axis=-1).reshape(data_shape)
    print("Generated Data (class representation):")
    print(generated_data_class)

latent_dim = 100
data_shape = (10, 10)

critic = build_critic(data_shape)
generator = build_generator(latent_dim, data_shape)
wgan = build_wgan(generator, critic)

train_wgan(wgan, generator, critic, latent_dim, epochs=10, batch_size=32, data_shape=data_shape, 
           critic_interval=5, generator_interval=1)
