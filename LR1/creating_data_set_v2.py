# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:03:49 2021

@author: Andrey
"""
import os
import h5py
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import pickle

# Входной data_set (данные). Я смешал все чтобы было больше кадров и потом уже
# делил на 3 части
input_test = r'C:\Users\Andrey\Downloads\kaggle-one-shot-pokemon\all'
# Дополнительный data_set. Можно использовать, но без него тоже работает
input_add = r'C:\Users\Andrey\Downloads\kaggle-one-shot-pokemon\add'
# Использовать дополнительный data_set. 1 - да, 0 - нет
flag_add_data = 0
# Название выходного файла с data_set
output = "dataset.hdf5"

im_w, im_h = 300, 300 # Ширина и высота к которой приводится модель

if flag_add_data:
    # Можно звать самому какие классы брать
    classes = {25: 'Pikachu', 1: 'Bulbasaur', 4: 'Charmander',
               201: 'Anoun', 6: 'Charizard', 100: 'Voltorb',
               133: 'Eevee', 130: 'Gyarados', 52: 'Meowth',
               111: 'Rhydon'}
    classes_list = [x for x in classes.keys()]
    count_cl = len(classes_list)
else:
    count_cl = 10 # Количество классов
    classes_list = list()

print('Количество классов для обучения %d' % count_cl)

def append_m(array, count, base_m):
    for value in array:
        base_m.append((os.path.join(input_test, value), count))

def find_name(data):
    # Получение класса из файла
    name = data.split('.')[0]
    if not name.isdigit():
        name = name.split('-')[0]
        if not name.isdigit():
            return
    return int(name)

def load(fName):
    # Формирование изображение для записи в файл и вдальнейшем для обучения
    im = Image.open(fName).convert('RGB')
    im = np.array(im.resize((300, 300)))
    im = Image.fromarray(im)
    im = img_to_array(im)
    # im = np.array(im)
    return im

# Деление всех файлов на классы
data_dir = dict()

for data in os.listdir(input_test):
    name = find_name(data)
    if name is not None:
        if name in data_dir:
            data_dir[name].append(data)
        else:
            data_dir[name] = [data]

# Деление файлов на выбранных классов на тренировочный, тестовый
# и проверочный набор данных
train_image = []
val_image = []
test_image = []
for kn in range(count_cl):
    if flag_add_data:
        lab = classes_list[kn]
        values = data_dir[lab]
    else:
        lab, values = max(data_dir.items(), key=lambda x: len(x[1]))
        print(lab, len(values))
        classes_list.append(lab)

    val_len = int(len(values)*0.1)
    if val_len == 0: val_len = 1
    test_len = int(len(values)*0.1)
    if test_len == 0: test_len = 1
    train_len = len(values) - test_len - val_len
    train_set = slice(0, train_len)
    val_set   = slice(train_len, train_len+val_len)
    test_set  = slice(train_len+val_len,len(values))

    for lng, slc in ([train_image, train_set],
                     [val_image, val_set],
                     [test_image, test_set]):
        append_m(values[slc], kn, lng)

    data_dir[lab] = []

# Добавление файлов из добавочных если нужно
if flag_add_data:
    for data in os.listdir(input_add):
        name = find_name(data)
        if name is not None and name in classes_list:
            train_image.append((os.path.join(input_add, data),
                                classes_list.index(name)))

# Запись файла с классами, можно использовать для тестов
with open('model.classes','wb') as f:
    pickle.dump(classes_list, f)

# Расчет количества всех кадров в каждом массиве
all_image = len(train_image) + len(val_image) + len(test_image)
train_len = len(train_image)
val_len   = len(val_image)
test_len  = len(test_image)
train_set = slice(0, train_len)
val_set   = slice(train_len, train_len+val_len)
test_set  = slice(train_len+val_len, all_image)

# Запись данных в файл
hdf5 = h5py.File(output, mode='w')

for lng, slc, nm in ([train_len, train_set, 'train'],
                     [val_len, val_set, 'val'],
                     [test_len, test_set, 'test']):
    hdf5.create_dataset("%s_img" % nm,(lng, im_h, im_w, 3), np.int8)
    hdf5.create_dataset("%s_labels" % nm,(lng,), np.int8)


for nm, slc in (['train', train_image],
                ['val', val_image],
                ['test', test_image]):
    for i, image in enumerate(slc):
        fName, label = image
        try:
            hdf5["%s_img" % nm][i, ...] = load(fName)
            hdf5["%s_labels" % nm][i, ...] = label
        except Exception as e:
            print(nm, fName,' : ', e)

hdf5.close()
print("Done %d+%d+%d=%d images processed!" % (train_len,val_len,test_len,
                                              all_image))
