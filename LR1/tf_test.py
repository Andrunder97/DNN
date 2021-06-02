# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:49:20 2021

@author: Andrey
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Путь к данным
hdf5_path = r"C:\Users\Andrey\Downloads\dataset.hdf5"
# Путь к модели
weights = hdf5_path.replace('dataset','model')

def plot_image(i, predictions_array, true_label, img):
    true_label, img = int(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("Результат модели {} {:2.0f}% \n  Действительность {} {:2.0f}%".format(predicted_label,
                                  100*np.max(predictions_array),
                                  true_label, 100*predictions_array[true_label]),
                                  color=color)
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    # x_labels = predictions_array > 0.1
    # plt.xticks(np.arange(50)[x_labels])
    plt.yticks([])
    thisplot = plt.bar(range(count_cl), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

# Загрузки файла с данными
hdf5 = h5py.File(hdf5_path,'r')
val_img = hdf5["val_img"][...]
val_labels = hdf5["val_labels"][...]
count_cl = max(val_labels) + 1 # Количество классов

hdf5.close()
arr = val_img/255.0

# Загрузка модели
new_model = tf.keras.models.load_model(weights)
probability_model = keras.Sequential([new_model, keras.layers.Softmax()])
predictions = probability_model.predict(arr)

num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], val_labels, val_img)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], val_labels)
plt.tight_layout()
plt.show()