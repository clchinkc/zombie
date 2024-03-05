import keras
import numpy as np
import tensorflow as tf
from keras import layers


# Define the generator
def build_generator(data_shape, num_classes, num_layers=1, filter_size=16, dropout_rate=0.25):
    noise_input = layers.Input(shape=data_shape + (1,))
    timestep_input = layers.Input(shape=(1,))

    timestep_dense = layers.Dense(data_shape[0] * data_shape[1], use_bias=False)(timestep_input)
    timestep_reshaped = layers.Reshape(data_shape + (1,))(timestep_dense)
    merged_input = layers.Add()([noise_input, timestep_reshaped])

    x = layers.Flatten()(merged_input)
    x = layers.Dense((data_shape[0] - 2 * num_layers - 2) * (data_shape[1] - 2 * num_layers - 2) * filter_size)(x)
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
def build_critic_pacgan(data_shape, num_classes, num_layers=1, filter_size=32, dropout_rate=0.25, m=2):
    data_input = layers.Input(shape=data_shape + (num_classes * m,))
    timestep_input = layers.Input(shape=(m,))
    
    timestep_dense = layers.Dense(int(np.prod(data_shape)) * num_classes * m, use_bias=False)(timestep_input)
    timestep_reshaped = layers.Reshape(data_shape + (num_classes * m,))(timestep_dense)
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
    alpha_shape = [batch_size, 1, 1, 1]
    alpha = tf.random.uniform(shape=alpha_shape, minval=0, maxval=1)
    
    interpolated_images = (real_images * alpha) + (fake_images * (1 - alpha))
    
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
    generator_optimizer = keras.optimizers.Nadam(learning_rate=0.00005)
    
    critic.compile(loss=wasserstein_loss, optimizer=critic_optimizer)
    generator.compile(loss=wasserstein_loss, optimizer=generator_optimizer)

    return critic, generator, critic_optimizer, generator_optimizer

# Define training steps
def critic_training_step(critic, generator, batch_size, num_classes, critic_optimizer, real_images, real_labels, m):
    noise = tf.random.normal([batch_size * m, data_shape[0], data_shape[1], 1])
    fake_labels = tf.random.uniform([batch_size * m, 1], 0, num_classes, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        fake_images = generator([noise, fake_labels], training=True)
        fake_images_packed = tf.reshape(fake_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
        
        selected_indices = tf.random.shuffle(tf.range(start=0, limit=batch_size * m, delta=1))
        shuffled_real_images = tf.gather(real_images, selected_indices, axis=0)
        real_images_packed = tf.reshape(shuffled_real_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
        shuffled_real_labels = tf.gather(real_labels, selected_indices, axis=0)
        real_labels_packed = tf.reshape(shuffled_real_labels, (batch_size, m))
        
        real_output = critic([real_images_packed, real_labels_packed], training=True)
        fake_output = critic([fake_images_packed, real_labels_packed], training=True)
        
        critic_real_loss = tf.reduce_mean(fake_output)
        critic_fake_loss = -tf.reduce_mean(real_output)
        gp = gradient_penalty(batch_size, real_images_packed, fake_images_packed, critic, real_labels_packed, strength=10.0)
        
        critic_loss = critic_real_loss + critic_fake_loss + gp
    
    critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables)) if critic_gradients else None
    
    return critic_loss, critic_real_loss, critic_fake_loss

def generator_training_step(generator, critic, batch_size, num_classes, generator_optimizer, m):
    noise = tf.random.normal([batch_size * m, data_shape[0], data_shape[1], 1])
    fake_labels = tf.random.uniform([batch_size * m, 1], 0, num_classes, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        generated_images = generator([noise, fake_labels], training=True)
        generated_images_packed = tf.reshape(generated_images, (batch_size, data_shape[0], data_shape[1], num_classes * m))
        packed_labels = tf.reshape(fake_labels, (batch_size, m))

        gen_output = critic([generated_images_packed, packed_labels], training=True)
        generator_loss = -tf.reduce_mean(gen_output)
    
    generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables)) if generator_gradients else None
    
    return generator_loss

def train_wgan(generator, critic, critic_optimizer, generator_optimizer, data_shape, num_classes, epochs, batch_size, critic_interval, generator_interval, m=2):
    for epoch in range(epochs):
        critic_losses, critic_real_losses, critic_fake_losses, generator_losses = [], [], [], []
        for _ in range(critic_interval):
            real_labels = tf.random.uniform([batch_size * m, 1], 0, num_classes, dtype=tf.int32)
            real_images = tf.one_hot(tf.random.uniform([batch_size * m, *data_shape], 0, num_classes, dtype=tf.int32), depth=num_classes)
            critic_loss, critic_real_loss, critic_fake_loss = critic_training_step(
                critic, generator, batch_size, num_classes, critic_optimizer, real_images, real_labels, m
            )
            critic_losses.append(critic_loss)
            critic_real_losses.append(critic_real_loss)
            critic_fake_losses.append(critic_fake_loss)
            
        for _ in range(generator_interval):
            generator_loss = generator_training_step(
                generator, critic, batch_size, num_classes, generator_optimizer, m
            )
            generator_losses.append(generator_loss)
        
        avg_critic_loss = np.mean([loss.numpy() for loss in critic_losses])
        avg_critic_real_loss = np.mean([loss.numpy() for loss in critic_real_losses])
        avg_critic_fake_loss = np.mean([loss.numpy() for loss in critic_fake_losses])
        avg_generator_loss = np.mean([loss.numpy() for loss in generator_losses])
        print(f"Epoch {epoch + 1}/{epochs} \t[ Critic Loss: {avg_critic_loss:.4f}, Critic Real Loss: {avg_critic_real_loss:.4f}, Critic Fake Loss: {avg_critic_fake_loss:.4f}, Generator Loss: {avg_generator_loss:.4f} ]")

    sample_noise = np.random.normal(0, 1, (1, data_shape[0], data_shape[1], 1))
    sample_labels = np.array([np.random.randint(0, num_classes)]).reshape((1, 1))
    generated_images = generator.predict([sample_noise, sample_labels], verbose=0)
    generated_class = np.argmax(generated_images, axis=-1).reshape(data_shape)
    print("Generated Data (class representation):")
    print(generated_class)

data_shape = (10, 10)
latent_dim = 10
num_classes = 4

generator = build_generator(data_shape, num_classes)
generator.summary()
critic = build_critic_pacgan(data_shape, num_classes)
critic.summary()
critic, generator, critic_optimizer, generator_optimizer = build_wgan(generator, critic)

# Train the model
train_wgan(generator, critic, critic_optimizer, generator_optimizer, data_shape, num_classes, epochs=10, batch_size=16, critic_interval=5, generator_interval=1)

# Use tf.Data.dataset, fit after defining custom training loop
