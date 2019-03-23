# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:50:27 2019

@author: shj_k
"""

from keras.layers import Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose, TimeDistributed, Flatten, Reshape
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), 
                          input_shape=(5,128, 128 ,1))) 

model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(units=64, return_sequences=True))

model.add(TimeDistributed(Reshape((8, 8, 1))))
model.add(TimeDistributed(UpSampling2D((2,2))))
model.add(TimeDistributed(Conv2D(16, (3,3), activation='relu', padding='same')))
model.add(TimeDistributed(UpSampling2D((2,2))))
model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')))
model.add(TimeDistributed(UpSampling2D((2,2))))
model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same')))
model.add(TimeDistributed(UpSampling2D((2,2))))
model.add(TimeDistributed(Conv2D(1, (3,3), padding='same')))

model.compile(optimizer='adam', loss='mse')
print ("2")
data = np.load(r"C:\Users\shj_k\Desktop\Project\walking.npy")
print (data.shape)
(x_train,x_test) = train_test_split(data)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

history = model.fit(x_train, x_train,
                epochs=100,
                batch_size=1,
                shuffle=False,
                validation_data=(x_test, x_test))

encoded_imgs = model.predict(x_test)
decoded_imgs = model.predict(encoded_imgs)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.summary()