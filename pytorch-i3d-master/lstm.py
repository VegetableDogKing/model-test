import numpy as np
import pandas as pd
import os
from talos import Scan
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

directory_path = "pytorch-i3d-master/features"

file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.npy')]
labels = np.loadtxt('/home/vegetabledogkingm/Desktop/model test/pytorch-i3d-master/list.txt')

data = []

# Load data and labels
for file_path in file_paths:
    loaded_data = np.load(file_path)
    # Reshape data to (2, 1024)
    reshaped_data = loaded_data.reshape((2, 1024))
    data.append(reshaped_data)

# Convert data and labels to numpy arrays
X = np.array(data)
y = labels

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

def lstm_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(2, 1024)))

    model.add(LSTM(units=params['lstm_units'], return_sequences=True,
                   kernel_regularizer='l2', recurrent_regularizer='l2', activity_regularizer='l2'))

    model.add(Dense(units=params['fc_units'], activation='relu',
                    kernel_regularizer='l2', activity_regularizer='l2'))

    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=lr_normalizer(params['lr'], Adam)),
                  loss=params['loss'],
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=(x_val, y_val), verbose=0,
                        callbacks=[early_stopper(params['epochs'], mode='moderate')])

    return history, model

p = {
    'lstm_units': (32, 256, 8),
    'fc_units': (16, 128, 8),
    'dropout': (0, 0.5, 0.1),
    'lr': (0.001, 0.01, 0.1),
    'loss': ['mean_squared_error', 'binary_crossentropy'],
    'epochs': (50, 2000, 50),
    'batch_size': (50, 2000, 50),
}

scan_object = Scan(x=X,
                   y=y,
                   model=lstm_model,
                   params=p,
                   experiment_name='lstm_hyperparameter_optimization',
                   val_split=0.2,
                   print_params=True,
                   random_method='uniform_mersenne')

# Access the results
print(scan_object.data)