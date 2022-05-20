#                 -----------------------------------------------
#                 |     CREATING A MACHINE LEARNING MODEL       |
#                 -----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tks
from tensorflow.keras.datasets import mnist

(x,y),(x_val,y_val)=mnist.load_data()      # this automatically divides data into training and validation set.

# Prepare data for training, while dealing with images these 3 preparations are necessary.

x=x.reshape(60000,784)                     #  1) Flattening of images
x_val=x_val.reshape(10000,784)             #

x,x_val=x/255,x_val/255                    #  2) Normalizing data

y=tks.utils.to_categorical(y,10)           #  3) Categorical encoding of values
y_val=tks.utils.to_categorical(y_val,10)   #

# Creating the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

""""Dense means all neurons are connected with neurons in other layer
    units is number of neurons for particular layer
    activation is activation function used
    input_shape is number of variables given as input"""

model = Sequential()                                                   # instantiating model

model.add(Dense(units=512,activation='relu',input_shape=(784,)))       # add input layer
model.add(Dense(units=512,activation='relu'))                          # add hidden layer
model.add(Dense(units=10,activation='softmax'))                        # add output layer

model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
h=model.fit(x,y,epochs=5,verbose=1,validation_data=(x_val,y_val))

# Results of model.fit can be accessed using h.history
print(h.history)

#                   *******************************
#                   *     IMPLEMENTING A CNN      *
#                   *******************************

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization,)

model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation="softmax"))
