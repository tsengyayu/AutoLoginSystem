# from extra_keras_datasets import emnist
# from keras_preprocessing.image import ImageDataGenerator
# from keras.utils import np_utils
# import os
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# from keras.utils import np_utils, plot_model
# import tensorflow as tf
# (input_train, target_train), (input_test, target_test) = emnist.load_data(type='balanced')
# print(len(input_train), len(target_train), len(input_test), len(target_test))
# x_train = input_train.reshape(112800, 1, 28, 28)/255
# x_test = input_test.reshape(18800, 1, 28, 28)/255
# y_train = np_utils.to_categorical(target_train)
# y_test = np_utils.to_categorical(target_test)
#
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(47, activation='softmax'))
# print(model.summary())
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)
#
# # Test
# loss, accuracy = model.evaluate(x_test, y_test)
# print('Test:')
# print('Loss: %s\nAccuracy: %s' % (loss, accuracy))


import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras import models
from keras.utils import img_to_array
from keras.utils import load_img

epochs = 30
img_rows = None
img_cols = None
digits_in_img = 4
x_list = list()
y_list = list()
x_train = list()
y_train = list()
x_test = list()
y_test = list()

def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])


img_filenames = os.listdir('training')

for img_filename in img_filenames:
    if '.png' not in img_filename:
        continue
    img = load_img('training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)

y_list = keras.utils.to_categorical(y_list, num_classes=10)
x_train, x_test, y_train, y_test = train_test_split(x_list, y_list)

# if os.path.isfile('cnn_model.h5'):
#     model = models.load_model('cnn_model.h5')
#     print('Model loaded from file.')
# else:
model = models.Sequential()
model.add(
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols // digits_in_img, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(10, activation='softmax'))
print('New model created.')

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), batch_size=5, epochs=epochs, verbose=1,
          validation_data=(np.array(x_test), np.array(y_test)))

loss, accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

model.save('cnn_model.h5')

