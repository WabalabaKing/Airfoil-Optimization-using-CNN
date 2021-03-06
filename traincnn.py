# -*- coding: utf-8 -*-
"""TrainCNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D6AM_9GsV-0zKK1y8g3IDxOw1RrE9Hs1
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import scipy.io
import time

tf.test.gpu_device_name()

data = scipy.io.loadmat('Train.mat')
dataV = scipy.io.loadmat('Valid.mat')
dataT = scipy.io.loadmat('Test.mat')
X,y = data['datax'],data['datay']
Xv,yv = dataV['datax'],dataV['datay']
Xt,yt = dataT['datax'],dataT['datay']
#Train for CD, load this
y_train= np.reshape(y[:,2],(len(y[:,1]),-1))
y_val = np.reshape(yv[:,2],(len(yv[:,1]),-1))
y_test = np.reshape(yt[:,2],(len(yt[:,1]),-1))
#Train for CL load this
#y = y[:,0]
#yv = yv[:,0]
#yt = yt[:,0]


X_train= X.reshape(-1,128,128,1)
X_val= Xv.reshape(-1,128,128,1)
X_test = Xt.reshape(-1,128,128,1)
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 1

batch_size = 8
learning_rate = 1e-5
num_epochs = 650

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as k

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Flatten, Dense, Activation, BatchNormalization, Concatenate

model = Sequential()

model.add(Input(shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
model.add(Conv2D(10, (13, 13), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(20, (7, 7), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(40, (7, 7), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(80, (5, 5), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(720, activation='swish'))
model.add(Flatten())
model.add(Dense(500, activation='swish'))
model.add(Dense(1, activation='swish'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),  #SGD(lr=0.005, momentum=0.9), #Adam(learning_rate = 2e-4),
                        loss = tf.keras.losses.MSE, #'categorical_crossentropy'
                        metrics = ['mse']) #'acc'

# Stop training when the val_loss has stopped decreasing for 5 epochs.
es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                       restore_best_weights=True, verbose=1)

STEP_SIZE_TRAIN = len(X_train)//batch_size
STEP_SIZE_VALID = len(X_val)//batch_size

# reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.3,
                                  patience = 3,
                                  min_lr = 1e-8,
                                  mode = 'min',
                                  verbose = 1)

# Save the model with the minimum validation loss
checkpoint_cb = ModelCheckpoint("./best_model.h5",
                                    save_best_only=True,
                                    monitor = 'val_loss',
                                    mode='min')
    
history = model.fit(x=X_train,y=y_train, validation_data=(X_val,y_val), 
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[es, reduce_lr],
                    batch_size= batch_size,epochs= num_epochs)

y_pred = model.predict(X_test.reshape(-1,128,128,1))
y_pred

r2_score(y_test,y_pred)

plt.figure(figsize=(15,10))
pre = [x for x in y_pred]
test = [x for x in y_test]
#plt.figure()
plt.scatter(pre, test,s=1)
#plt.scatter(y_pred, y_test,s=1)

plt.xlabel('Predicted Cl/Cd Ratio')
plt.ylabel('Actual Cl/Cd Ratio')

plt.title(' Test & Predicted confusion matrix')
plt.show()

import pickle
model.save('PredictCMby1')

!zip -r /content/PredictCMby1.zip /content/PredictCMby1/
from google.colab import files
files.download('/content/PredictCMby1.zip')