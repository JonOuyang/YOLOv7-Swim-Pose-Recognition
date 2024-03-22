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

import math

#The following code section was for ensuring that Jupyter Notebook was running on GPU instead of CPU for faster training

# Check if GPU is available and visible to TensorFlow
gpu_available = tf.config.list_physical_devices('GPU')
print("GPU Available:", gpu_available)
# Check which device TensorFlow is using for computations (CPU or GPU)
device_name = tf.test.gpu_device_name()
if device_name == '':
    device_name = '/CPU:0'
print("TensorFlow is using:", device_name)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


num_classes = 2
xt = np.load('x2_test.npy')
yt = np.load('y2_test.npy')
yt = keras.utils.to_categorical(yt, num_classes)
tl = np.argmax(yt, axis=1)

class CustomPrintCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        
        model.evaluate(xt, yt, verbose=1)
        predicted_labels=np.argmax(model.predict(xt), axis=1)
        #tl = yt
        confusion_mat = confusion_matrix(tl, predicted_labels)
        print(confusion_mat)
        print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')


num_classes = 2

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x2_test.npy')
y_test = np.load('y2_test.npy')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = keras.Sequential([
        keras.Input(shape=(32,12,3)),
        layers.Dense(40, activation='relu'),
        layers.Dense(40, activation='relu'),
        layers.Flatten(),
        layers.Dense(2, activation="softmax"),
    ])

def relu_bn(inputs: Tensor) -> Tensor:
    
    relu = layers.LeakyReLU(alpha=0.01)(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    y = Dropout(0.1)(y)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(32, 12, 3))
    num_filters = 100
    
    t = BatchNormalization()(inputs)
    #k_s, 4, 4, stride 2
    t = Conv2D(kernel_size=4,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    kernel_regularizer=l2(0.001)
    #ernel_regularizer=l2(0.001)
    t = relu_bn(t)
    
    #2, 4, 4, 2
    #num_blocks_list = [2, 5, 5, 2]
    #enhanced: 5, 9, 9, 5
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters*=2
    
    t = AveragePooling2D(2)(t)
    #t = MaxPooling2D(pool_size=(2, 2))(t)
    t = Flatten()(t)
    #p = Dense(32, activation='relu')(t)
    outputs = Dense(2, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        #optimizer = Adam(learning_rate=0.0001),
        #optimizer = Adam(learning_rate=4e-7),
        optimizer=Adam(learning_rate=3e-6),
        #optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
model = create_res_net()

#Working
#odel.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=3e-5), metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])


model.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=1e-5), metrics="accuracy")


#model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics="accuracy")


model.summary()

#model = create_res_net()
#78% acc.
#model = tf.keras.models.load_model('testModel65')

for i in range(2):

    k = 10  # For example, using 5-fold cross-validation

    # Create a KFold object
    kf = KFold(n_splits=k, shuffle=True)

    # Initialize lists to store training and validation results for each fold

    #model.summary()

    # Loop through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(x_test)):
        print(f"Training fold {fold + 1}")
        x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        
        model_checkpoint = ModelCheckpoint(filepath = f'model_{fold}.h5', monitor='val_loss')

        
        # Train the model on this fold
        history = model.fit(
            x=x_train_fold,
            y=y_train_fold,
            epochs=20,
            verbose=0,
            batch_size=32,
            validation_data=(x_val_fold, y_val_fold),
            callbacks=[model_checkpoint]
            #class_weight=class_weights
            #validation_data=(x_test, y_test)
        )
        model.evaluate(x_test, y_test, verbose=1)
        predicted_labels=np.argmax(model.predict(xt), axis=1)
        tl = np.argmax(yt, axis=1)
        #tl = ytR
        confusion_mat = confusion_matrix(tl, predicted_labels)
        print(confusion_mat)
        print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')


"""
%%time
accuracyArray = []
loops = 0
while True:
    try:
        #Model
        model = create_res_net()

        #model.compile(loss="categorical_crossentropy", optimizer = Adam(learning_rate=3e-6), metrics="accuracy")
        model.compile(loss="categorical_crossentropy", optimizer = 'sgd', metrics="accuracy")
        accuracyArray = []
        for i in range(3):

            k = 10  # For example, using 5-fold cross-validation

            # Create a KFold object
            kf = KFold(n_splits=k, shuffle=True)

            # Initialize lists to store training and validation results for each fold

            #model.summary()

            # Loop through each fold
            for fold, (train_index, val_index) in enumerate(kf.split(x_test)):
                print(f"Training fold {fold + 1}")
                x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                model_checkpoint = ModelCheckpoint(filepath = f'model_{fold}.h5', monitor='val_loss')


                # Train the model on this fold
                history = model.fit(
                    x=x_train_fold,
                    y=y_train_fold,
                    epochs=20,
                    verbose=0,
                    batch_size=32,
                    validation_data=(x_val_fold, y_val_fold),
                    #class_weight=class_weights
                    #validation_data=(x_test, y_test)
                    callbacks=[model_checkpoint]
                )
                model.evaluate(x_test, y_test, verbose=1)
                predicted_labels=np.argmax(model.predict(xt), axis=1)
                tl = np.argmax(yt, axis=1)
                #tl = ytR
                confusion_mat = confusion_matrix(tl, predicted_labels)
                print(confusion_mat)
                print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')
                accuracyArray.append((np.trace(confusion_mat))/np.sum(confusion_mat))
            if any(num > 0.75 for num in accuracyArray):
                break
        loops +=1
    except:
        pass
    if any(num > 0.75 for num in accuracyArray):
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print(f'Model trained a total of {loops} times!')
        print("Training Complete!")
        break
    else:
        clear_output(wait=True)
"""

#model.save('testModel60')
test = tf.keras.models.load_model('model_2.h5')
#test = tf.keras.models.load_model('testModel65')
#test.summary()
test.evaluate(x_test, y_test, verbose=1)
predicted_labels=np.argmax(test.predict(xt), axis=1)
tl = np.argmax(yt, axis=1)
#tl = yt
confusion_mat = confusion_matrix(tl, predicted_labels)
print(confusion_mat)
print(f'Model Accuracy based on testing data: {(np.trace(confusion_mat))/np.sum(confusion_mat)}')

#test.save('testModel94')
