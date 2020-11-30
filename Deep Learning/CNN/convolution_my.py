# -*- coding: utf-8 -*-
import tensorflow as tf
tf.__version__

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,#because of this our pixel value will come in 0 to 1
                                   shear_range = 0.2,#random transection
                                   zoom_range = 0.2,#random Zoom
                                   horizontal_flip = True)#it will filp horizontally so dont found same image in diff batches


test_datagen = ImageDataGenerator(rescale = 1./255) #value become between 0 to 1


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),#size of images expected for our cnn model
                                                 batch_size = 32,#here batch goes into cnn after updating weights
                                                 class_mode = 'binary')#we have cats and dog so binary


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')



from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

cnn=Sequential()


cnn.add(Convolution2D(filters=32,kernel_size=3,input_shape=[64,64,3],activation="relu",padding="same"))

cnn.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

cnn.add(Flatten())

cnn.add(Dense(units=128,activation='relu'))

cnn.add(Dense(units=1,activation='sigmoid'))

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

cnn.fit_generator(training_set,
                  steps_per_epoch =8000,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 2000)
print(len(tf.config.experimental.list_physical_devices('GPU')))
