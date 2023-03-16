# Setting warnings off
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import layers, Input
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Lambda
print(tf. __version__, keras.__version__) 

df = pd.read_excel("ralstonia_high.xlsx")

inputFeatures = range(42)
targetVariable = ['ralstonia']
df_Repeat = df.iloc[np.repeat(np.arange(len(df)), len(df))]
df_tile = df.iloc[np.tile(np.arange(len(df)), len(df))]

x_Repeat = df_Repeat[inputFeatures]
y_Repeat = df_Repeat[targetVariable].to_numpy()
x_tile = df_tile[inputFeatures]
y_tile = df_tile[targetVariable].to_numpy()

y_diff = y_Repeat - y_tile
print(y_Repeat.shape, y_tile.shape, y_diff.shape)

from sklearn.model_selection import train_test_split

x_Repeat_train, x_Repeat_test, y_Repeat_train, y_Repeat_test = train_test_split(x_Repeat,y_Repeat,test_size=0.2)
x_tile_train, x_tile_test, y_tile_train, y_tile_test = train_test_split(x_tile,y_tile,test_size=0.2)

y_diff_train = y_Repeat_train - y_tile_train
y_diff_test = y_Repeat_test - y_tile_test


def Twin(X1, X2, YY, X1_val, X2_val, YY_val, hLayers, epochs,verbose=3):
    
    inputs1 = keras.layers.Input(shape=(X1.shape[1] + X2.shape[1]))
    
    modelLayer1 = inputs1
    
    for hSize in hLayers:
        modelLayer1 = keras.layers.Dense(hSize, activation='relu')(modelLayer1)
        
    distanceLayer = keras.layers.Dense(1)(modelLayer1)
    
    model = keras.models.Model(inputs1, distanceLayer)
    
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss='mse')
    print(model.summary())

    X = np.concatenate((X1, X2), axis=1)
    X_val = np.concatenate((X1_val, X2_val), axis=1)
    loss = model.fit(X,YY, validation_data=(X_val, YY_val),
                     batch_size=1024, epochs=epochs,verbose=verbose)
    # Save the model
    model.save('tf_fork4.h5')
    return model, loss

hLayers = [60]
model, loss = Twin(x_Repeat_train,x_tile_train,y_diff_train,
                      x_Repeat_test, x_tile_test, y_diff_test,
                      hLayers, epochs=100, verbose=1)
pyplot.plot(loss.history['loss'])