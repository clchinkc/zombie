
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Add,
    Attention,
    AveragePooling1D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
    MultiHeadAttention,
    Multiply,
    Reshape,
)
from torch import layer_norm

# Load data
prices_df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
prices_df['Price'] = scaler.fit_transform(prices_df['Close'].values.reshape(-1, 1))

# Create training and testing datasets
train_data, test_data = train_test_split(prices_df['Price'], test_size=0.2, shuffle=False)

# Create time steps for training and testing datasets
def create_dataset(dataset, time_step=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_step - 1):
        data_X.append(dataset[i:(i + time_step)])
        data_Y.append(dataset[i + time_step])
    return np.array(data_X), np.array(data_Y)

time_step = 60
train_X, train_Y = create_dataset(train_data, time_step)
test_X, test_Y = create_dataset(test_data, time_step)

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# Use tf.data.Dataset to load and preprocess data
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_dataset = train_dataset.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
test_dataset = test_dataset.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)


def multi_filter_block(inputs, filters=32, kernel_sizes=[3, 7, 14, 30, 60, 120, 240, 360]):
    # Apply multiple filters of different sizes
    filter_layers = []
    excess = 0
    for size in kernel_sizes:
        if size > time_step:
            size = kernel_sizes[excess]
            excess += 1
        conv_layer = Conv1D(filters=filters, kernel_size=size, activation='relu', padding='causal')(inputs)
        filter_layers.append(conv_layer)
    concat1 = Concatenate()(filter_layers)
    norm_concat1 = LayerNormalization()(concat1)
    return norm_concat1

# CNN + LSTM Model
def cnn_lstm_model():
    inputs = Input(shape=(time_step, 1))
    noise = GaussianNoise(0.01)(inputs)
    
    norm_concat1 = multi_filter_block(noise)
    
    # Apply LSTM layers
    lstm1 = LSTM(64, return_sequences=True, dropout=0.25)(norm_concat1)
    layer_norma1 = LayerNormalization()(lstm1)
    lstm2 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma1)
    layer_norma2 = LayerNormalization()(lstm2)
    lstm3 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma2)
    layer_norma3 = LayerNormalization()(lstm3)
    lstm4 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma3)
    layer_norma4 = LayerNormalization()(lstm4)
    
    # Apply CNN layers
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(norm_concat1)
    layer_normb1 = LayerNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb1)
    layer_normb2 = LayerNormalization()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb2)
    layer_normb3 = LayerNormalization()(conv3)
    conv4 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb3)
    layer_normb4 = LayerNormalization()(conv4)
    
    # Add full residual connection
    add = Add()([layer_norma1, layer_norma2, layer_norma3, layer_norma4])
    multiply = Multiply()([layer_normb1, layer_normb2, layer_normb3, layer_normb4])
    
    concat = Concatenate()([add, multiply])
    
    flatten = Flatten()(concat)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def cnn_lstm_model_1():
    inputs = Input(shape=(time_step, 1))
    noise = GaussianNoise(0.01)(inputs)
    
    norm_concat1 = multi_filter_block(noise)
    
    # Apply LSTM layers
    lstm1 = LSTM(64, return_sequences=True, dropout=0.25)(norm_concat1)
    layer_norma1 = LayerNormalization()(lstm1)
    lstm2 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma1)
    layer_norma2 = LayerNormalization()(lstm2)
    lstm3 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma2)
    layer_norma3 = LayerNormalization()(lstm3)
    lstm4 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norma3)
    layer_norma4 = LayerNormalization()(lstm4)
    
    # Apply CNN layers
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(norm_concat1)
    layer_normb1 = LayerNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb1)
    layer_normb2 = LayerNormalization()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb2)
    layer_normb3 = LayerNormalization()(conv3)
    conv4 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_normb3)
    layer_normb4 = LayerNormalization()(conv4)
    
    # Add scaled residual connections
    resa1 = Lambda(lambda x: x * 0.125)(layer_norma1)
    resa2 = Lambda(lambda x: x * 0.125)(layer_norma2)
    resa3 = Lambda(lambda x: x * 0.125)(layer_norma3)
    resa4 = Lambda(lambda x: x * 0.125)(layer_norma4)
    
    resb1 = Lambda(lambda x: x * 0.125)(layer_normb1)
    resb2 = Lambda(lambda x: x * 0.125)(layer_normb2)
    resb3 = Lambda(lambda x: x * 0.125)(layer_normb3)
    resb4 = Lambda(lambda x: x * 0.125)(layer_normb4)
    
    # Add full residual connection
    add = Add()([resa1, resa2, resa3, resa4, resb1, resb2, resb3, resb4])
    
    flatten = Flatten()(add)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def cnn_model(kernel_sizes=[3, 7, 14, 30, 60, 120, 240, 360]):
    inputs = Input(shape=(time_step, 1))
    noise = GaussianNoise(0.01)(inputs)

    norm_concat1 = multi_filter_block(noise)

    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(norm_concat1)
    layer_norm1 = LayerNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm1)
    layer_norm2 = LayerNormalization()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm2)
    layer_norm3 = LayerNormalization()(conv3)
    conv4 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm3)
    layer_norm4 = LayerNormalization()(conv4)
    conv5 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm4)
    layer_norm5 = LayerNormalization()(conv5)
    conv6 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm5)
    layer_norm6 = LayerNormalization()(conv6)
    conv7 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm6)
    layer_norm7 = LayerNormalization()(conv7)
    conv8 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm7)
    layer_norm8 = LayerNormalization()(conv8)
    
    # Add full residual connection
    add = Multiply()([layer_norm1, layer_norm2, layer_norm3, layer_norm4, layer_norm5, layer_norm6, layer_norm7, layer_norm8])
    
    flatten = Flatten()(add)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


class TPALayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TPALayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(
            filters=1, 
            kernel_size=input_shape[1], 
            activation='sigmoid', 
            padding='same',
            name='tpa_conv'
        )
        self.layer_norm = LayerNormalization()
        self.multiply = Multiply()
        super(TPALayer, self).build(input_shape)

    def call(self, x):
        x_transposed = tf.transpose(x, [0, 2, 1])
        conv = self.conv(x_transposed)
        layer_norm = self.layer_norm(conv)
        tpa_transposed = tf.transpose(layer_norm, [0, 2, 1])
        multiply = self.multiply([x, tpa_transposed])
        return tpa_transposed
    
    def get_config(self):
        config = super(TPALayer, self).get_config()
        config.update({
            'conv': self.conv,
        })
        return config

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout_rate)
        self.layer_norm = LayerNormalization()
        self.multiply = Multiply()
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention = self.multi_head_attention(inputs, inputs)
        layer_norm = self.layer_norm(attention)
        multiply = self.multiply([inputs, layer_norm])
        return layer_norm

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

def lstm_multihead_attention_model():
    inputs = Input(shape=(time_step, 1))
    noise = GaussianNoise(0.01)(inputs)
    lstm1 = LSTM(64, return_sequences=True, dropout=0.25)(noise)
    layer_norm1 = LayerNormalization()(lstm1)
    lstm2 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm1)
    layer_norm2 = LayerNormalization()(lstm2)
    lstm3 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm2)
    layer_norm3 = LayerNormalization()(lstm3)
    lstm4 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm3)
    layer_norm4 = LayerNormalization()(lstm4)
    lstm5 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm4)
    layer_norm5 = LayerNormalization()(lstm5)
    lstm6 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm5)
    layer_norm6 = LayerNormalization()(lstm6)
    lstm7 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm6)
    layer_norm7 = LayerNormalization()(lstm7)
    lstm8 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm7)
    layer_norm8 = LayerNormalization()(lstm8)
    
    add1 = Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4, layer_norm5, layer_norm6, layer_norm7, layer_norm8])

    # Temporal pooling attention
    tpa1 = TPALayer()(add1)
    
    # Multi-head attention
    multihead1 = MultiHeadAttentionLayer(num_heads=1, key_dim=64, dropout_rate=0.25)(add1)
    
    add2 = Add()([add1, tpa1, multihead1])
    
    # Flatten and output
    flatten = Flatten()(add2)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


import tensorflow_addons as tfa


def nas_rnn_model():
    inputs = tf.keras.Input(shape=(time_step, 1))
    noise = tf.keras.layers.GaussianNoise(0.01)(inputs)
    
    LSTMCell1 = tfa.rnn.NASCell(64)
    rnn1 = tf.keras.layers.RNN(LSTMCell1, return_sequences=True)
    lstm1 = rnn1(noise)

    layer_norm1 = tf.keras.layers.LayerNormalization()(lstm1)

    LSTMCell2 = tfa.rnn.NASCell(64)
    rnn2 = tf.keras.layers.RNN(LSTMCell2, return_sequences=True)
    lstm2 = rnn2(layer_norm1)

    layer_norm2 = tf.keras.layers.LayerNormalization()(lstm2)
    
    LSTMCell3 = tfa.rnn.NASCell(64)
    rnn3 = tf.keras.layers.RNN(LSTMCell3, return_sequences=True)
    lstm3 = rnn3(layer_norm2)
    
    layer_norm3 = tf.keras.layers.LayerNormalization()(lstm3)
    
    LSTMCell4 = tfa.rnn.NASCell(64)
    rnn4 = tf.keras.layers.RNN(LSTMCell4, return_sequences=True)
    lstm4 = rnn4(layer_norm3)
    
    layer_norm4 = tf.keras.layers.LayerNormalization()(lstm4)
    
    add = tf.keras.layers.Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4])

    flatten = tf.keras.layers.Flatten()(add)
    outputs = tf.keras.layers.Dense(1)(flatten)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


import tensorflow_wavelets.Layers.DWT as DWT


def wavelet_model():
    inputs = Input(shape=(time_step, 1))
    inputs_reshaped = Reshape((time_step, 1, 1))(inputs)
    wavelet = DWT.DWT()(inputs_reshaped)
    wavelet_reshaped = Reshape((time_step, 2))(wavelet)
    flatten = Flatten()(wavelet_reshaped)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(32, activation='relu')(dense1)
    outputs = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_block(inputs, filters, kernel_size, padding='causal'):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
    x = LayerNormalization()(x)
    
    shortcut = Conv1D(filters=filters, kernel_size=1, padding=padding)(inputs)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def dense_block(inputs, num_layers, growth_rate, kernel_size, padding='causal'):
    x = inputs
    for i in range(num_layers):
        # Convolutional layer
        x = Conv1D(filters=growth_rate, kernel_size=kernel_size, padding=padding)(x)
        # Layer normalization
        x = LayerNormalization()(x)
        # ReLU activation
        x = Activation('relu')(x)
        # Convolutional layer
        x = Conv1D(filters=growth_rate, kernel_size=kernel_size, padding=padding)(x)
        # Layer normalization
        x = LayerNormalization()(x)
        # Concatenate with previous layers
        inputs = Concatenate()([inputs, x])
    return inputs

def densenet_resnet_model():
    inputs = Input(shape=(time_step, 1))

    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(inputs)

    # First dense block
    dense1 = dense_block(conv1, num_layers=4, growth_rate=32, kernel_size=3, padding='causal')

    # First residual block
    res_block1 = resnet_block(dense1, filters=64, kernel_size=3)

    # Second dense block
    dense2 = dense_block(res_block1, num_layers=4, growth_rate=32, kernel_size=3, padding='causal')

    # Second residual block
    res_block2 = resnet_block(dense2, filters=64, kernel_size=3)

    flatten = Flatten()(res_block2)
    outputs = Dense(1)(flatten)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Transformer-based model
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, d_model, num_heads, dropout_rate):
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = tf.keras.layers.Dense(d_model, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return out2

def build_transformer_model(time_step, d_model, num_heads, num_layers, dropout_rate): # 118.51069440971501
    inputs = tf.keras.Input(shape=(time_step, 1))
    x = PositionalEncoding(time_step, d_model)(inputs)

    for _ in range(num_layers):
        x = transformer_encoder(x, d_model, num_heads, dropout_rate)

    # x = Dense(1)(x)
    # outputs = tf.squeeze(x, axis=-1)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Variational LSTM Model

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        mean, variance = inputs
        
        if training:
            epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
            return mean + K.exp(variance / 2) * epsilon
        elif not training:
            return mean

class PredictionSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, variance = inputs
        return mean

class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        y_true, y_pred, mean_layer, variance_layer = inputs

        reconstruction_loss = K.mean(K.square(y_true - y_pred), axis=-1)
        kl_loss = -0.5 * K.mean(1 + variance_layer - K.square(mean_layer) - K.exp(variance_layer), axis=[-1, -2])
        return K.mean(reconstruction_loss + kl_loss)

def var_lstm_model():
    inputs = Input(shape=(time_step, 1))
    y_true = Input(shape=(1,))
    
    noise = GaussianNoise(0.01)(inputs)
    lstm1 = LSTM(64, return_sequences=True, dropout=0.25)(noise)
    layer_norm1 = LayerNormalization()(lstm1)
    
    mean_layer = Dense(units=1)(layer_norm1)
    variance_layer = Dense(units=1)(layer_norm1)
    
    sampling = Sampling()([mean_layer, variance_layer])
    
    flatten = Flatten()(sampling)
    outputs = Dense(1)(flatten)

    vae_loss_layer = VAELossLayer()([y_true, outputs, mean_layer, variance_layer])

    training_model = Model(inputs=[inputs, y_true], outputs=outputs)
    training_model.add_loss(vae_loss_layer)
    
    prediction_model = Model(inputs=inputs, outputs=outputs)

    return training_model, prediction_model


model = cnn_lstm_model() # 151.24902135240558 0.00817540567368269 22862.9921875
# model = cnn_model() # 146.26684775725644 0.006160466931760311 21392.03515625
# model = lstm_multihead_attention_model() # 179.6202153271338 0.05479753017425537 32182.181640625
# model = nas_rnn_model() # 155.13566803010383 0.001908295089378953 24062.21484375
# model = wavelet_model() # 140.8948050487553 0.0010008609388023615 19850.70703125
# model = densenet_resnet_model() # 155.51024688443073 0.012486668303608894 24160.541015625
# model = build_transformer_model(time_step, d_model=64, num_heads=4, num_layers=2, dropout_rate=0.25) # 83.796415175186 0.022571461275219917 6581.54248046875
# model, prediction_model = var_lstm_model() # 64.90182741723712 0.01319837011396885 16638.669921875


model.summary()

initial_learning_rate = 0.001
decay_steps = 1000
alpha = 0.1

# Define the learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha
)

# Define the Adam optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(train_dataset, epochs=20, validation_data=test_dataset, verbose=1)

# Predict and evaluate the model
train_predict = model.predict(train_dataset)
test_predict = model.predict(test_dataset)

train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))

print("Train RMSE:", np.sqrt(mean_squared_error(train_Y, train_predict)))
print("Train model score:", model.evaluate(train_X, train_Y, verbose=0))
print("Test RMSE:", np.sqrt(mean_squared_error(test_Y, test_predict)))
print("Test model score:", model.evaluate(test_X, test_Y, verbose=0))

# New function to make rolling predictions
def rolling_predict(model, initial_input, num_predictions):
    predictions = []
    current_input = initial_input

    for _ in range(num_predictions):
        # Make a prediction using the current input
        predicted_value = model.predict(current_input, verbose=0)

        # Add the predicted value to the list of predictions
        predictions.append(predicted_value[0])

        # Update the current input by removing the oldest value and appending the predicted value
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1, 0] = predicted_value

    return np.array(predictions)

# Calculate the initial input for rolling predictions
initial_input = test_X[0].reshape(1, time_step, 1)

# Make rolling predictions
test_predict_rolling = rolling_predict(model, initial_input, len(test_X))

# Inverse transform the rolling predictions
test_predict_rolling = scaler.inverse_transform(test_predict_rolling)

# Calculate RMSE for the rolling predictions
print("Rolling Test RMSE:", np.sqrt(mean_squared_error(test_Y, test_predict_rolling)))
print("Rolling Test model score:", model.evaluate(test_X, test_predict_rolling, verbose=0))

# Update the plotting section with the rolling predictions
plt.figure(figsize=(12, 6))
plt.plot(prices_df['Close'], label='Historical Prices')

train_range = np.arange(time_step, len(train_data) - 1)
test_range = np.arange(len(train_data) + time_step, len(prices_df) - 1)

train_dates = prices_df.iloc[train_range, :].index
test_dates = prices_df.iloc[test_range, :].index

plt.plot(test_dates, test_predict, label='Test Predictions')
plt.plot(test_dates, test_predict_rolling, label='Rolling Test Predictions', linestyle='--')  # Add rolling predictions
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()


# https://www.kaggle.com/code/bryanb/stock-prices-forecasting-with-lstm/notebook
