

import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
data = data.dropna()
data = data[['Close']]

min_price = data['Close'].min()
max_price = data['Close'].max()

scaled_data = data.apply(lambda x: (x - min_price) / (max_price - min_price))

# Define the hyperparameters
num_epochs = 10
batch_size = 512
learning_rate = 0.00001
latent_dim = 10
window_size = 14

# Prepare the input data
def create_dataset(data, window_size):
    x_data = []
    y_data = []
    for i in range(len(data) - window_size):
        x_data.append(data.iloc[i:i + window_size].values)
        y_data.append(data.iloc[i + window_size].values)
    return np.array(x_data), np.array(y_data)

x_data, y_data = create_dataset(scaled_data, window_size)

# Define the generator network
def build_generator():
    input_stock = tf.keras.layers.Input(shape=(window_size, 1))
    input_noise = tf.keras.layers.Input(shape=(window_size, latent_dim - 1))

    input_data = tf.keras.layers.Concatenate(axis=2)([input_stock, input_noise])
    
    x = tf.keras.layers.LSTM(256, return_sequences=True)(input_data)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(x)
    x = tf.keras.layers.Dense(1, activation='tanh')(x)

    model = tf.keras.Model(inputs=[input_stock, input_noise], outputs=x)
    return model

# Define the discriminator network
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(window_size + 1, 1), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Define the generator loss function
def generator_loss(y_true, fake_output):
    return tf.keras.losses.BinaryCrossentropy()(y_true, fake_output)

# Define the discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define the optimizer for both the generator and discriminator networks
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate * 5)

# Instantiate the generator and discriminator networks
generator = build_generator()
discriminator = build_discriminator()

# Define the input data for the generator
input_data = tf.keras.layers.Input(shape=(window_size, 1))
input_noise = tf.keras.layers.Input(shape=(window_size, latent_dim - 1))
fake_data = generator([input_data, input_noise])

# Define the input data for the discriminator
real_data = tf.keras.layers.Input(shape=(window_size + 1, 1))
real_output = discriminator(real_data)
fake_output = discriminator(tf.keras.layers.concatenate([input_data, fake_data], axis=1))

# Compile the generator and discriminator models
generator_model = tf.keras.models.Model(inputs=[input_data, input_noise], outputs=fake_data)
generator_model.compile(loss=generator_loss, optimizer=generator_optimizer)

discriminator_model = tf.keras.models.Model(inputs=[real_data, input_data, input_noise], outputs=[real_output, fake_output])
discriminator_model.compile(loss=[discriminator_loss, discriminator_loss], optimizer=discriminator_optimizer)

# Define the training loop
for epoch in range(num_epochs):
    for i in range(len(scaled_data) // batch_size):
        # Train the discriminator
        for j in range(10):
            index_start = i * batch_size
            index_end = (i + 1) * batch_size
            real_data_batch = scaled_data[index_start:index_end + 1].values.reshape(batch_size + 1, -1, 1)[:-1]
            x_batch = x_data[index_start:index_end]
            noise = np.random.normal(0, 1, (batch_size, window_size, latent_dim - 1))
            fake_data_batch = generator.predict([x_batch, noise])

            # Concatenate x_batch and real/fake_data_batch along the time axis
            real_data_with_history = np.concatenate([x_batch, real_data_batch], axis=1)
            fake_data_with_history = np.concatenate([x_batch, fake_data_batch], axis=1)

            d_loss_total, d_loss_real, d_loss_fake = discriminator_model.train_on_batch([real_data_with_history, x_batch, noise], [np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            d_loss = d_loss_real + d_loss_fake

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, window_size, latent_dim - 1))
        g_loss = generator_model.train_on_batch([x_batch, noise], np.ones((batch_size, 1)))

    # Print the losses at the end of each epoch
    print(f"Epoch {epoch + 1}: Discriminator Loss = {d_loss_total}, Generator Loss = {g_loss}")


import matplotlib.pyplot as plt
import numpy as np

past_14_days = x_data[-1:]  # Assuming the last window in x_data contains the most recent 14 days
predicted_prices = []

# Predict the next 14 days
for i in range(window_size):
    noise = np.random.normal(0, 1, (1, window_size, latent_dim - 1))
    predicted_price = generator.predict([past_14_days, noise])

    # Add the predicted price to the list and update the input data for the next prediction
    predicted_prices.append(predicted_price[0, -1])
    past_14_days = np.append(past_14_days[:, 1:], predicted_price, axis=1)

# Rescale the predicted prices
rescaled_predicted_prices = np.array(predicted_prices) * (max_price - min_price) + min_price

# Plot the predicted prices
plt.plot(rescaled_predicted_prices)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.yscale('linear')
plt.title('Predicted Stock Prices for the Next 14 Days')
plt.show()



# Plot historical data and all the simulated price trajectories using a line plot
plt.plot(data.index, data['Close'].values, label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=window_size + 1, freq='D')[1:], rescaled_predicted_prices, label='Simulated Trajectories')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Trajectories')
plt.legend()
plt.show()