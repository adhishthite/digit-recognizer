''' CONVOLUTIONAL NEURAL NETWORK MODEL'''

# IMPORTS
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# HYPERPARAMETERS
batch_size = 128
num_classes = 10
epochs = 15

# Input Image Dimensions
img_rows, img_cols = 28, 28

# Splitting the Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data (refer- Keras Documentation Examples)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Image Normalization
x_train /= 255
x_test /= 255

# One Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# KERAS MODEL

model = Sequential()

# First Layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Hidden Layers (Convolution)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

# DNN Classifier Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Compiling the Model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest loss:', score[0])
print('\nTest accuracy:', score[1])

model.save('cnn-mnist-model.h5')


''' MODEL SUMMARY '''

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)           (None, 26, 26, 32)        320
# _________________________________________________________________
# conv2d_2 (Conv2D)           (None, 24, 24, 64)        18496
# _________________________________________________________________
# conv2d_3 (Conv2D)           (None, 22, 22, 64)        36928
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 11, 11, 64)        0
# _________________________________________________________________
# dropout_1 (Dropout)         (None, 11, 11, 64)        0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 7744)              0
# _________________________________________________________________
# dense_1 (Dense)             (None, 128)               991360
# _________________________________________________________________
# dense_2 (Dense)             (None, 64)                8256
# _________________________________________________________________
# dropout_2 (Dropout)         (None, 64)                0
# _________________________________________________________________
# dense_3 (Dense)             (None, 10)                650
# =================================================================
# Total params: 1,056,010
# Trainable params: 1,056,010
# Non-trainable params: 0
# _________________________________________________________________