# -*- coding: utf-8 -*-


import h5py
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import utils as np_utils
#from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np



# Файл в который сохранили data_set
hdf5_path = r"C:\Users\Andrey\Downloads\dataset.hdf5"

# Чтение всех полученных данных
datafile = h5py.File(hdf5_path,'r')
train_img = datafile["train_img"][...]
train_labels = datafile["train_labels"][...]
test_img = datafile["test_img"][...]
test_labels = datafile["test_labels"][...]
validate_img = datafile["val_img"][...]
validate_labels = datafile["val_labels"][...]
# Закрытие файла
datafile.close()

# Количество классов
count_cl = max(train_labels) + 1

# Получение размера изображения
img_size = train_img.shape[1:]
print('Размер изображение:', img_size)

# =============================================================================
# plt.figure(figsize=(10,10))
# for i in range(len(test_img)):
#     plt.figure(figsize=(10,10))
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_img[i])
#     plt.tight_layout()
#     plt.show()
# =============================================================================

# Показ часть тренировочных данных
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[25+i])
    plt.xlabel(train_labels[25+i])

plt.tight_layout()
plt.show()

# Модель и слои из примера лабораторной работы (обучается довольно быстро)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=img_size),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(count_cl)
])

# =============================================================================
# # Модель сложнее
# model = keras.Sequential()
# model.add(Conv2D(16, kernel_size=(4), activation=tf.nn.relu, input_shape=img_size))
# model.add(MaxPooling2D(pool_size=(4)))
# model.add(Flatten())
# model.add(Dropout(0.3))
# model.add(Dense(count_cl, activation='softmax'))
# =============================================================================

# =============================================================================
# std_init = 0.01
# input_shape = (300, 300, 1)
# left_input = tf.keras.layers.Input(input_shape)
# right_input = tf.keras.layers.Input(input_shape)
#
# w_init = tf.keras.initializers.RandomNormal(0,std_init)
# b_init = tf.keras.initializers.RandomNormal(0.5,std_init)
# regul = tf.keras.regularizers.l2(0)
#
# model = tf.keras.models.Sequential()
# model.add(Conv2D(64, (10, 10), activation='relu', input_shape=img_size,
#                  kernel_initializer=w_init, kernel_regularizer=regul))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (7, 7), activation='relu',
#                  kernel_initializer=w_init, kernel_regularizer=regul,bias_initializer=b_init))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (4, 4), activation='relu',
#                  kernel_initializer=w_init, kernel_regularizer=regul,bias_initializer=b_init))
# model.add(MaxPooling2D())
# model.add(Conv2D(256, (4, 4), activation='relu',
#                  kernel_initializer=w_init, kernel_regularizer=regul,bias_initializer=b_init))
# model.add(Dropout(0))
# model.add(Flatten())
# model.add(Dense(4096, activation="sigmoid",
#                                  kernel_initializer=tf.keras.initializers.RandomNormal(0,0.1),
#                                   kernel_regularizer=regul,
#                                   bias_initializer=tf.keras.initializers.RandomNormal(0,0.1)))
# model.add(Dropout(0))
# encoded_l = model(left_input)
# encoded_r = model(right_input)
# merged = tf.math.abs(encoded_l - encoded_r)
# prediction = tf.keras.layers.Dense(11, activation='sigmoid',
#                                   kernel_initializer=tf.keras.initializers.RandomNormal(0,0.1),
#                                   kernel_regularizer=regul,
#                                   bias_initializer=tf.keras.initializers.RandomNormal(0,0.1))(merged)
# =============================================================================

# =============================================================================
# model = keras.Sequential()
# #1
# model.add(Conv2D(32, kernel_size=3, input_shape=img_size))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# #model.add(MaxPooling2D(pool_size=2))
#
# #2
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
#
# #3
# model.add(Conv2D(64, kernel_size=3))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
#
# #4
# model.add(Conv2D(64, kernel_size=3))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# #model.add(MaxPooling2D(pool_size=2))
#
# #5
# model.add(Conv2D(128, kernel_size=3))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
#
# #6
# model.add(Conv2D(128, kernel_size=3))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
#
# #7
# model.add(Flatten())
# model.add(Dense(count_cl, activation='softmax'))
#
# =============================================================================

model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_img/255.0, train_labels, # Тренировочные данные
          batch_size=96, # Деление дата_Сета на пакеты
          shuffle=True,  # Перетасовка кадров в дата сете
          epochs=1000, # Сколько раз нейронная сеть пройдет через весь дата сет
          validation_data=((test_img/255.0), test_labels), # Тестовые данные
          callbacks=[EarlyStopping(min_delta=0.001, patience=5)]) # Чтобы модель не переобучилась ставим выход

# model.fit(train_img/255.0, train_labels, epochs=50)

# Запуск проверки на проверочных данных
loss, acc = model.evaluate(validate_img/255.0, validate_labels)
print("Loss: %.3f \nAccuracy: %.3f" % (loss, acc))

# Сохранение модели в файл
output = hdf5_path.replace('dataset','model')
tf.keras.models.save_model(model, output)

print('Model saved to %s' % output)


