import numpy as np
import tensorflow as tf
from keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Reshape,
    Softmax,
)
from keras.models import Model, Sequential
from keras.optimizers import RMSprop


# Define the generator
def build_generator(latent_dim, data_shape, num_classes):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,))

    label_embedding = Dense(latent_dim, use_bias=False)(label_input)
    merged_input = Add()([noise_input, label_embedding])

    x = Reshape((5, 5, -1))(merged_input)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation="elu")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation="elu")(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(1, 1), padding='same', activation="elu")(x)
    x = Softmax(axis=-1)(x)

    return Model(inputs=[noise_input, label_input], outputs=x)

# Define the critic
def build_critic(data_shape, num_classes):
    image_input = Input(shape=(*data_shape, num_classes))
    label_input = Input(shape=(1,))

    label_embedding = Dense(int(np.prod(data_shape)) * num_classes, use_bias=False)(label_input)
    label_embedding = Reshape((*data_shape, num_classes))(label_embedding)
    
    merged_input = Add()([image_input, label_embedding])

    x = Conv2D(128, kernel_size=(3, 3), padding='valid', activation="elu", kernel_constraint=lambda w: tf.clip_by_value(w, -0.01, 0.01))(merged_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3, 3), padding='valid', activation="elu", kernel_constraint=lambda w: tf.clip_by_value(w, -0.01, 0.01))(x)  # Additional Conv2D as per reference
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1, activation='linear')(x)

    return Model(inputs=[image_input, label_input], outputs=x)

# Update GAN building function
def build_wgan(generator, critic):
    # Wasserstein loss function
    def wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)
    critic.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))
    critic.trainable = False

    noise_input, label_input = generator.input
    generated_image = generator.output
    critic_output = critic([generated_image, label_input])

    gan = Model([noise_input, label_input], critic_output)
    gan.compile(loss=wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))

    return gan

# Update training procedure
def train_wgan(gan, generator, critic, latent_dim, epochs, batch_size, data_shape, num_classes, critic_interval=2, generator_interval=1):
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))

    for epoch in range(epochs):
        c_loss_real = 0
        c_loss_fake = 0
        g_loss_total = 0
        
        for _ in range(critic_interval):
            noise = tf.random.normal([batch_size, latent_dim])
            labels = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            gen_imgs = generator.predict([noise, labels], verbose=0)
            real_imgs = tf.one_hot(tf.random.uniform([batch_size, *data_shape], minval=0, maxval=num_classes, dtype=tf.int32), depth=num_classes)
            real_labels = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)

            real_dataset = tf.data.Dataset.from_tensor_slices((real_imgs, real_labels, valid)).map(lambda x, y, z: ((x, y), z), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            fake_dataset = tf.data.Dataset.from_tensor_slices((gen_imgs, labels, fake)).map(lambda x, y, z: ((x, y), z), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            critic.trainable = True
            c_real_loss = critic.fit(real_dataset, epochs=critic_interval, verbose=0)
            c_fake_loss = critic.fit(fake_dataset, epochs=critic_interval, verbose=0)
            c_loss_real += c_real_loss.history['loss'][-1]
            c_loss_fake += c_fake_loss.history['loss'][-1]


        for _ in range(generator_interval):
            critic.trainable = False
            noise = tf.random.normal([batch_size, latent_dim])
            labels = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            noise_dataset = noise_dataset = tf.data.Dataset.from_tensor_slices((noise, labels, valid)).map(lambda x, y, z: ((x, y), z), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            g_loss = gan.fit(noise_dataset, epochs=generator_interval, verbose=0)
            g_loss_total += g_loss.history['loss'][-1]

        c_loss_real_avg = c_loss_real / critic_interval
        c_loss_fake_avg = c_loss_fake / critic_interval
        g_loss_avg = g_loss_total / generator_interval

        print(f"Epoch {epoch}/{epochs} [Critic: real loss: {c_loss_real_avg:.4f}, fake loss: {c_loss_fake_avg:.4f}] [Generator loss: {g_loss_avg:.4f}]")

    sample_noise = np.random.normal(0, 1, (1, latent_dim))
    sample_label = np.array([0]).reshape((1, 1))
    generated_data = generator.predict([sample_noise, sample_label], verbose=0)
    generated_data_class = np.argmax(generated_data, axis=-1).reshape(data_shape)
    print("Generated Data (class representation):")
    print(generated_data_class)

latent_dim = 100
data_shape = (10, 10)
num_classes = 4

generator = build_generator(latent_dim, data_shape, num_classes)
generator.summary()
critic = build_critic(data_shape, num_classes)
critic.summary()
wgan = build_wgan(generator, critic)

train_wgan(wgan, generator, critic, latent_dim, epochs=10, batch_size=32, data_shape=data_shape, num_classes=num_classes,
            critic_interval=2, generator_interval=1)
