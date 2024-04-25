# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Softmax

def load_model(model_path):
    global probability_model 
    model = models.load_model(model_path)
    probability_model = Sequential([model, Softmax()])

def predict_instability(frame):
    
    predictions = probability_model.predict(frame)
    
    predicted = np.argmax(predictions[0])
    
    return (predicted, predictions)