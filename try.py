import numpy as np
import tensorflow as tf
from keras import layers
from keras.constraints import MinMaxNorm
from keras.models import Model
from keras.optimizers import RMSprop


# Define the generator
def build_generator(latent_dim, data_shape, num_classes, num_layers=1, filter_size=16, dropout_rate=0.25):
    noise_input = layers.Input(shape=(latent_dim,))
    timestep_input = layers.Input(shape=(1,))

    timestep_dense = layers.Dense(latent_dim, use_bias=False)(timestep_input)
    merged_input = layers.Add()([noise_input, timestep_dense])

    x = layers.Dense((data_shape[0] - 2 * num_layers - 2) * (data_shape[1] - 2 * num_layers - 2) * filter_size)(merged_input)
    x = layers.Reshape((data_shape[0] - 2 * num_layers - 2, data_shape[1] - 2 * num_layers - 2, filter_size))(x)
    x = layers.ELU()(x)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        x = layers.Conv2DTranspose(filter_size, kernel_size=(3, 3), padding='valid')(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2DTranspose(num_classes, kernel_size=(3, 3), padding='valid', activation="softmax")(x)

    model = Model(inputs=[noise_input, timestep_input], outputs=x)
    return model

# Define the critic
def build_critic(data_shape, num_classes, num_layers=1, filter_size=32, dropout_rate=0.25):
    data_input = layers.Input(shape=(*data_shape, num_classes))
    timestep_input = layers.Input(shape=(1,))
    
    timestep_dense = layers.Dense(int(np.prod(data_shape)) * num_classes, use_bias=False)(timestep_input)
    timestep_reshaped = layers.Reshape((*data_shape, num_classes))(timestep_dense)
    merged_input = layers.Add()([data_input, timestep_reshaped])
    
    x = layers.GaussianNoise(0.1)(merged_input)
    x = layers.ELU()(x)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='valid', kernel_constraint=MinMaxNorm(min_value=-0.01, max_value=0.01, rate=1.0))(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=[data_input, timestep_input], outputs=x)
    return model

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
    valid = -np.random.uniform(low=0.9, high=1.0, size=batch_size)
    fake = np.random.uniform(low=0.9, high=1.0, size=batch_size)

    for epoch in range(epochs):
        c_loss_real, c_loss_fake, g_loss_total = 0, 0, 0
        
        for _ in range(critic_interval):
            noise = tf.random.normal([batch_size, latent_dim])
            labels_noise = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            gen_imgs = generator.predict([noise, labels_noise], verbose=0)
            labels_sample = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            real_imgs = tf.one_hot(tf.random.uniform([batch_size, *data_shape], minval=0, maxval=num_classes, dtype=tf.int32), depth=num_classes)

            real_dataset = tf.data.Dataset.from_tensor_slices(((real_imgs, labels_sample), valid)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            fake_dataset = tf.data.Dataset.from_tensor_slices(((gen_imgs, labels_noise), fake)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            critic.trainable = True
            c_real_loss = critic.fit(real_dataset, epochs=1, verbose=0)
            c_fake_loss = critic.fit(fake_dataset, epochs=1, verbose=0)
            c_loss_real += c_real_loss.history['loss'][-1]
            c_loss_fake += c_fake_loss.history['loss'][-1]


        for _ in range(generator_interval):
            critic.trainable = False
            noise = tf.random.normal([batch_size, latent_dim])
            labels_noise = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            noise_dataset = noise_dataset = tf.data.Dataset.from_tensor_slices(((noise, labels_noise), valid)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            g_loss = gan.fit(noise_dataset, epochs=1, verbose=0)
            g_loss_total += g_loss.history['loss'][-1]

        c_loss_real_avg = c_loss_real / critic_interval
        c_loss_fake_avg = c_loss_fake / critic_interval
        g_loss_avg = g_loss_total / generator_interval

        print(f"Epoch {epoch+1}/{epochs} [Critic: real loss: {c_loss_real_avg:.4f}, fake loss: {c_loss_fake_avg:.4f}] [Generator loss: {g_loss_avg:.4f}]")

    sample_noise = np.random.normal(0, 1, (1, latent_dim))
    sample_label = np.array([0]).reshape((1, 1))
    generated_data = generator.predict([sample_noise, sample_label], verbose=0)
    generated_data_class = np.argmax(generated_data, axis=-1).reshape(data_shape)
    print("Generated Data (class representation):")
    print(generated_data_class)

data_shape = (10, 10)
latent_dim = np.prod(np.array(data_shape))
num_classes = 4

generator = build_generator(latent_dim, data_shape, num_classes)
generator.summary()
critic = build_critic(data_shape, num_classes)
critic.summary()
wgan = build_wgan(generator, critic)

train_wgan(wgan, generator, critic, latent_dim, epochs=20, batch_size=512, data_shape=data_shape, num_classes=num_classes,
            critic_interval=3, generator_interval=1)
