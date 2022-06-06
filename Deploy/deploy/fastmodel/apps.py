from django.apps import AppConfig
import html
import pathlib
from pathlib import Path
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers



class FastmodelConfig(AppConfig):
    
    MODEL_PATH = Path("model")
    MODEL = MODEL_PATH/'model3.h5'
    
    #default_auto_field = 'django.db.models.BigAutoField'
    DATA = MODEL_PATH/'input2.csv'


    train_features = pd.read_csv(DATA)
    train_features

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())

    def medium_model(norm):

        regularizer = 0.000001
        dropout = 0
        schedul = -0.0001



        lr = 0.001
  
        model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularizer) ),
        layers.Dropout(dropout),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(regularizer) ),
        layers.Dropout(dropout),
        layers.Dense(1)
        ])
        
        
        return model



    name = 'fastmodel'

    

    dnn = medium_model(normalizer)
    
    dnn.load_weights(MODEL)

    a = dnn.predict(train_features)

    def PRINT(a = a):
        return(a)    

