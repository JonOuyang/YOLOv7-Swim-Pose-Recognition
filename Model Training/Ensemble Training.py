#I genuinely don't even know what this code is for

# importing utility modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow import Tensor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import he_normal

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import datetime
import os
import matplotlib.pyplot as plt

#import sklearn
#from sklearn.metrics import confusion_matrix
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
# importing machine learning models for prediction
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier

num_classes=4
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x2_test.npy')
y_test = np.load('y2_test.npy')

#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

from sklearn.base import BaseEstimator, ClassifierMixin

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model):
        self.keras_model = keras_model
    
    def fit(self, X, y):
        # Your fit logic here
        pass
    
    def predict(self, X):
        predictions = self.keras_model.predict(X)
        return np.argmax(predictions, axis=-1)
    
    def predict_proba(self, X):
        return self.keras_model.predict(X)

model1 = tf.keras.models.load_model('testModel60')
model2 = tf.keras.models.load_model('testModel61')
model3 = tf.keras.models.load_model('testModel62')
model4 = tf.keras.models.load_model('testModel63')
model5 = tf.keras.models.load_model('testModel64')

model1 = KerasClassifierWrapper(model1)
model2 = KerasClassifierWrapper(model2)
model3 = KerasClassifierWrapper(model3)
model4 = KerasClassifierWrapper(model4)
model5 = KerasClassifierWrapper(model5)

# Assuming you have your training and testing data (x_train, y_train, x_test, y_test)

models = [model1, model2, model3, model4, model5]

predictions = []

for model in models:
    predictions.append(model.predict(x_test))
    
predictions = np.array(predictions)
   
#predictions = predictions.transpose((1, 0, 2))
    
# Making the final model using voting classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('model1', model1),
        ('model2', model2),
        ('model3', model3),
        ('model4', model4),
        ('model5', model5),
    ],
    voting='hard'
)

# training all the models on the training dataset (if not trained already)
#final_model.fit(x_train, y_train)

# predicting the output on the test dataset
voting_classifier.fit(x_train, y_train)
# printing log loss between actual and predicted values
ensemble_predictions = voting_classifier.predict(x_test)
ensemble_predictions = np.argmax(predictions, axis=-1)  # Hard voting

accuracy = accuracy_score(y_test, ensemble_predictions)
print(f'Ensemble Accuracy: {accuracy:.4f}')
