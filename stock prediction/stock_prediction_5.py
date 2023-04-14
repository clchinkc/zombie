
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
    GRU,
    LSTM,
    Activation,
    Add,
    Attention,
    AveragePooling1D,
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

# CNN + LSTM Model

def cnn_lstm_model():
    inputs = Input(shape=(time_step, 1))
    
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(inputs)
    layer_norm1 = LayerNormalization()(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(layer_norm1)
    layer_norm2 = LayerNormalization()(conv2)
    conv3 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='causal')(layer_norm2)
    layer_norm3a = LayerNormalization()(conv3)

    lstm1 = LSTM(32, return_sequences=True, dropout=0.25)(inputs)
    layer_norm1 = LayerNormalization()(lstm1)
    lstm2 = LSTM(64, return_sequences=True, dropout=0.25)(layer_norm1)
    layer_norm2 = LayerNormalization()(lstm2)
    lstm3 = LSTM(128, return_sequences=True, dropout=0.25)(layer_norm2)
    layer_norm3b = LayerNormalization()(lstm3)

    add = Add()([layer_norm3a, layer_norm3b])
    flatten = Flatten()(add)
    outputs = Dense(1)(flatten)

    model = Model(inputs=inputs, outputs=outputs)
    return model





def cnn_model(kernel_sizes=[3, 7, 14, 30, 60, 120]):
    inputs = Input(shape=(time_step, 1))

    # First set of convolutional layers
    conv_layers = []
    excess = 0
    for size in kernel_sizes:
        if size > time_step:
            size = kernel_sizes[excess]
            excess += 1
        conv_layer = Conv1D(filters=32, kernel_size=size, activation='relu', padding='causal')(inputs)
        norm_layer = LayerNormalization()(conv_layer)
        conv_layers.append(norm_layer)
    
    concat1 = Concatenate()(conv_layers)
    norm_concat1 = LayerNormalization()(concat1)

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
    
    # Add scaled residual connections
    res1 = Lambda(lambda x: x * 0.125)(layer_norm1)
    res2 = Lambda(lambda x: x * 0.125)(layer_norm2)
    res3 = Lambda(lambda x: x * 0.125)(layer_norm3)
    res4 = Lambda(lambda x: x * 0.125)(layer_norm4)
    res5 = Lambda(lambda x: x * 0.125)(layer_norm5)
    res6 = Lambda(lambda x: x * 0.125)(layer_norm6)
    res7 = Lambda(lambda x: x * 0.125)(layer_norm7)
    res8 = Lambda(lambda x: x * 0.125)(layer_norm8)
    
    # Add full residual connection
    add = Add()([res1, res2, res3, res4, res5, res6, res7, res8])
    
    flatten = Flatten()(add)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# LSTM Model
def lstm_model():
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
    
    add = Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4])
    
    flatten = Flatten()(add)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def gru_model():
    inputs = Input(shape=(time_step, 1))
    noise = GaussianNoise(0.01)(inputs)
    gru1 = GRU(32, return_sequences=True, dropout=0.25)(noise)
    layer_norm1 = LayerNormalization()(gru1)
    gru2 = GRU(64, return_sequences=True, dropout=0.25)(layer_norm1)
    layer_norm2 = LayerNormalization()(gru2)
    gru3 = GRU(128, return_sequences=True, dropout=0.25)(layer_norm2)
    layer_norm3 = LayerNormalization()(gru3)
    gru4 = GRU(256, return_sequences=True, dropout=0.25)(layer_norm3)
    layer_norm4 = LayerNormalization()(gru4)
    
    #add = Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4])
    
    flatten = Flatten()(layer_norm4)
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
    
    add1 = Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4])
    
    # Attention mechanism
    attention1 = MultiHeadAttention(num_heads=1, key_dim=64, dropout=0.25)(add1, add1)
    layer_norm5 = LayerNormalization()(attention1)
    multiply1 = Multiply()([add1, layer_norm5])
    add2 = Add()([add1, multiply1])
    
    attention2 = MultiHeadAttention(num_heads=1, key_dim=64, dropout=0.25)(add2, add2)
    layer_norm6 = LayerNormalization()(attention2)
    multiply2 = Multiply()([add2, layer_norm6])
    add3 = Add()([add2, multiply2])
    
    # Flatten and output
    flatten = Flatten()(add3)
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
        super(TPALayer, self).build(input_shape)

    def call(self, x):
        x_transposed = tf.transpose(x, [0, 2, 1])
        conv = self.conv(x_transposed)
        tpa = tf.multiply(conv, x_transposed)
        tpa_transposed = tf.transpose(tpa, [0, 2, 1])
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
        self.add = Add()
        super(MultiHeadAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention = self.multi_head_attention(inputs, inputs)
        layer_norm = self.layer_norm(attention)
        multiply = self.multiply([inputs, layer_norm])
        add = self.add([inputs, multiply])
        return add

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
    
    add1 = Add()([layer_norm1, layer_norm2, layer_norm3, layer_norm4])

    # Temporal pooling attention
    tpa1 = TPALayer()(add1)
    
    # Multi-head attention
    multihead1 = MultiHeadAttentionLayer(num_heads=1, key_dim=64, dropout_rate=0.25)(tpa1)
    
    # Flatten and output
    flatten = Flatten()(multihead1)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
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

def resnet_model():
    inputs = Input(shape=(time_step, 1))

    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(inputs)
    
    res_block1 = resnet_block(conv1, filters=64, kernel_size=3)
    res_block2 = resnet_block(res_block1, filters=128, kernel_size=3)
    
    flatten = Flatten()(res_block2)
    outputs = Dense(1)(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def densenet_model():
    inputs = Input(shape=(time_step, 1))

    # Initial convolutional layer
    conv0 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(inputs)

    # Dense blocks
    concat = conv0
    for i in range(4):
        # Convolutional layers within the block
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(concat)
        conv2 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(conv1)
        # Concatenate the input with the output of the convolutional layers
        concat = Concatenate()([concat, conv2])

    # Final convolutional layer
    conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='causal')(concat)

    flatten = Flatten()(conv3)
    outputs = Dense(1)(flatten)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def tcn_block(inputs, filters, kernel_size, dilation_rate, padding='causal'):
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding)(x)
    x = LayerNormalization()(x)
    
    shortcut = inputs if dilation_rate == 1 else None
    
    if shortcut is not None:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding=padding)(shortcut)
    
    x = Add()([x, shortcut]) if shortcut is not None else x
    x = Activation('relu')(x)
    return x

def tcn_model():
    inputs = Input(shape=(time_step, 1))

    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='causal')(inputs)
    
    tcn_block1 = tcn_block(conv1, filters=32, kernel_size=3, dilation_rate=1)
    tcn_block2 = tcn_block(tcn_block1, filters=64, kernel_size=3, dilation_rate=2)
    tcn_block3 = tcn_block(tcn_block2, filters=128, kernel_size=3, dilation_rate=4)
    
    concat = Concatenate()([tcn_block1, tcn_block2, tcn_block3])
    
    flatten = Flatten()(tcn_block3)
    
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


# model = cnn_lstm_model() # 146.95755226890842 0.006202755495905876 21592.046875
# model = cnn_model() # 140.725326545822 0.005043825600296259 19806.326171875
# model = lstm_model() # 160.23631797823094 0.012350289151072502 25647.783203125
# model = gru_model() # 150.34692755795803 0.005442610941827297 22595.126953125
model = nas_rnn_model() # 155.13566803010383 0.001908295089378953 24062.21484375
# model = lstm_multihead_attention_model() # 156.0758334958998 0.00855713989585638 24340.04296875
# model = wavelet_model() # 140.8948050487553 0.0010008609388023615 19850.70703125
# model = resnet_model() # 115.80924303888958 0.004751001019030809 26970.88671875
# model = densenet_model() # 165.03632384569838 0.001491556758992374 38292.29296875
# model = tcn_model() # 167.78402637485027 0.014250650070607662 27262.623046875
# model = build_transformer_model(time_step, d_model=64, num_heads=4, num_layers=2, dropout_rate=0.25) # 83.796415175186 0.022571461275219917 6581.54248046875
# model, prediction_model = var_lstm_model() # 128.90182741723712 0.01319837011396885 16638.669921875


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
