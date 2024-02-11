import keras
import numpy as np
import tensorflow as tf
from keras import layers


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

    model = keras.models.Model(inputs=[noise_input, timestep_input], outputs=x)
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
        x = layers.Conv2D(filter_size, kernel_size=(3, 3), padding='valid')(x)
        x = layers.ELU()(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='linear')(x)

    model = keras.models.Model(inputs=[data_input, timestep_input], outputs=x)
    return model

# Define WGAN with Gradient Penalty
def gradient_penalty(batch_size, real_images, fake_images, critic, labels, strength):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], minval=0, maxval=1)
    interpolated_images = real_images * alpha + fake_images * (1 - alpha)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        prediction = critic([interpolated_images, labels], training=True)
    gradients = tape.gradient(prediction, [interpolated_images])
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2) * strength
    return penalty

# Update GAN building function
def build_wgan(generator, critic):
    def wasserstein_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)
    critic_optimizer = keras.optimizers.Nadam(learning_rate=0.00005)
    critic.compile(loss=wasserstein_loss, optimizer=critic_optimizer)
    critic.trainable = False
    noise_input, label_input = generator.input
    generated_image = generator([noise_input, label_input])
    critic_output = critic([generated_image, label_input])
    gan = keras.models.Model([noise_input, label_input], critic_output)
    generator_optimizer = keras.optimizers.Nadam(learning_rate=0.00005)
    gan.compile(loss=wasserstein_loss, optimizer=generator_optimizer)
    return gan, critic_optimizer, generator_optimizer

# Update training procedure
def train_wgan(gan, generator, critic, critic_optimizer, generator_optimizer, latent_dim, epochs, batch_size, data_shape, num_classes, critic_interval, generator_interval):
    valid = -np.random.uniform(low=0.9, high=1.0, size=batch_size)
    fake = np.random.uniform(low=0.9, high=1.0, size=batch_size)
    
    for epoch in range(epochs):
        for _ in range(critic_interval):
            noise = tf.random.normal([batch_size, latent_dim])
            labels = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            fake_images = generator([noise, labels], training=True)
            real_images = tf.one_hot(tf.random.uniform([batch_size, *data_shape], minval=0, maxval=num_classes, dtype=tf.int32), depth=num_classes)
            critic.trainable = True
            with tf.GradientTape() as tape:
                fake_preds = critic([fake_images, labels], training=True)
                real_preds = critic([real_images, labels], training=True)
                c_loss_real = -tf.reduce_mean(real_preds * valid)
                c_loss_fake = tf.reduce_mean(fake_preds * fake)
                gp = gradient_penalty(batch_size, real_images, fake_images, critic, labels, strength=10.0)
                c_loss = c_loss_real + c_loss_fake + gp
            gradients_of_critic = tape.gradient(c_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables)) if gradients_of_critic else None

        for _ in range(generator_interval):
            noise = tf.random.normal([batch_size, latent_dim])
            labels = tf.random.uniform([batch_size, 1], minval=0, maxval=num_classes, dtype=tf.int32)
            critic.trainable = False
            with tf.GradientTape() as tape:
                generated_images = generator([noise, labels], training=True)
                gen_preds = critic([generated_images, labels], training=True)
                g_loss = -tf.reduce_mean(gen_preds)
            gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) if gradients_of_generator else None

        print(f"Epoch {epoch+1}/{epochs}, Critic Loss Real: {c_loss_real.numpy()}, Critic Loss Fake: {c_loss_fake.numpy()}, Total Critic Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}")

    sample_noise = np.random.normal(0, 1, (1, latent_dim))
    sample_labels = np.array([np.random.randint(0, num_classes)]).reshape((1, 1))
    generated_images = generator.predict([sample_noise, sample_labels], verbose=0)
    generated_class = np.argmax(generated_images, axis=-1).reshape(data_shape)
    print("Generated Data (class representation):")
    print(generated_class)


data_shape = (10, 10)
latent_dim = np.prod(np.array(data_shape))
num_classes = 4

generator = build_generator(latent_dim, data_shape, num_classes)
generator.summary()
critic = build_critic(data_shape, num_classes)
critic.summary()
wgan, critic_optimizer, generator_optimizer = build_wgan(generator, critic)

train_wgan(wgan, generator, critic, critic_optimizer, generator_optimizer, latent_dim, epochs=20, batch_size=512, data_shape=data_shape, num_classes=num_classes, critic_interval=5, generator_interval=2)
