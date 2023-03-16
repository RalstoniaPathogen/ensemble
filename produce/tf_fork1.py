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


def Siamese(X1, X2, YY, X1_val, X2_val, YY_val, hLayers, epochs,verbose=3):
    
    inputs1 = keras.layers.Input(shape=(X1.shape[1],))
    inputs2 = keras.layers.Input(shape=(X2.shape[1],))
    
    modelLayer1 = inputs1
    modelLayer2 = inputs2
    
    for hSize in hLayers:
        modelLayer1 = keras.layers.Dense(hSize, activation='relu')(modelLayer1)
        modelLayer2 = keras.layers.Dense(hSize, activation='relu')(modelLayer2)
        
    lModel1 = keras.layers.Dense(1)(modelLayer1)
    lModel2 = keras.layers.Dense(1)(modelLayer2)

    def l1distance(layersOneShot):
        return layersOneShot[0] - layersOneShot[1]
    
    distanceLayer = keras.layers.Lambda(l1distance)([lModel1, lModel2])
    
    model = keras.models.Model([inputs1, inputs2], distanceLayer)
    
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss='mse')

    loss = model.fit([X1, X2],YY, validation_data=([X1_val, X2_val], YY_val),
                     batch_size=1024, epochs=epochs,verbose=verbose)
    # Save the model
    model.save('tf_fork1.h5')
    return model, loss

hLayers = [30]
model, loss = Siamese(x_Repeat,x_tile,y_diff,
                      x_Repeat, x_tile, y_diff,
                      hLayers, epochs=100, verbose=1)
pyplot.plot(loss.history['loss'])