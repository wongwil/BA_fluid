# -*- coding: utf-8 -*-
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
import pickle
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from load_data import get_frames, get_label_names, balance_data, set_labels
from predict_instability import predict_instability
from calculate_accuracy import print_accuracy
from pathlib import Path
from data_augmentation import augment_data


MODEL_PATH = 'model_7_to_33_5x5f_32k_64k_128k_128d_256d_gpu_21ep_64bs_full_aug_14small_32wave_12drop.h5'
VIAL_FILE = '../labeling/1g/labels/vial{}_labels.csv'
LABELS_TO_USE = {'still', 'wave', 'nearDrops', 'smallDrops', 'drops', 'foam'}
VIAL_NUMBERS = list({7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33})
#VIAL_NUMBERS = list({9, 20, 21, 22, 27, 31, 33})
load_new_data = False
train_only_aug = True

set_labels(LABELS_TO_USE)
batch_size = 64
epochs = 21
label_names = get_label_names()
num_classes = len(label_names)

if load_new_data:
    path = VIAL_FILE.format(str(VIAL_NUMBERS[0]))
    (our_x, our_y) = get_frames(path=path, with_augmented=False)
    for i in range(len(VIAL_NUMBERS)):
        print(i)
        if i > 0:
            path = VIAL_FILE.format(str(VIAL_NUMBERS[i]))
            (x_data, y_data) = get_frames(path=path, with_augmented=False)
            our_x = np.concatenate((our_x, x_data))
            our_y = np.concatenate((our_y, y_data))
    pickle.dump(our_x, open('X.pkl', 'wb'))
    pickle.dump(our_y, open('Y.pkl', 'wb'))
else:
    our_x = pickle.load(open('X.pkl', 'rb'))
    our_y = pickle.load(open('Y.pkl', 'rb'))

#Balance and augment all data (train, test and validation)
if not train_only_aug:
    (our_x, our_y) = balance_data(our_x, our_y, limit_weights = [3, 4, 1, 1.4, 1.2, 1])
    
    (our_x, our_y) = augment_data(our_x, our_y)

#Change labels to one hot encoding
our_y_one_hot = to_categorical(our_y)

#First split train and test data, then split train and validation data
split_train_X,our_test_X,split_train_label,our_test_label = train_test_split(our_x, our_y_one_hot, test_size=0.2, random_state=13, stratify = our_y_one_hot)
train_X, valid_X, train_Y, valid_Y = train_test_split(split_train_X, split_train_label, test_size=0.2, random_state=17, stratify = split_train_label)

#Only balance and augment training data
if train_only_aug:
    train_Y = np.argmax(train_Y, axis=1)
    
    (train_X, train_Y) = balance_data(train_X, train_Y, limit_weights = [3, 4, 1, 1.4, 1.2, 1])
    
    (train_X, train_Y) = augment_data(train_X, train_Y)
    
    train_Y = to_categorical(train_Y)

if not Path(MODEL_PATH).is_file():
    fluid_model = Sequential()

    fluid_model.add(Conv2D(32, kernel_size=(5, 5), activation='linear', padding='same', input_shape=(140,180,1)))
    fluid_model.add(LeakyReLU(alpha=0.1))
    fluid_model.add(MaxPooling2D((2, 2),padding='same'))
    fluid_model.add(Dropout(0.25))
    fluid_model.add(Conv2D(64, (5, 5), activation='linear',padding='same'))
    fluid_model.add(LeakyReLU(alpha=0.1))
    fluid_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fluid_model.add(Dropout(0.25))
    fluid_model.add(Conv2D(128, (5, 5), activation='linear',padding='same'))
    fluid_model.add(LeakyReLU(alpha=0.1))
    fluid_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fluid_model.add(Dropout(0.4))
    fluid_model.add(Flatten())
    fluid_model.add(Dense(128, activation='linear'))
    fluid_model.add(LeakyReLU(alpha=0.1))
    fluid_model.add(Dropout(0.3))
    fluid_model.add(Dense(256, activation='linear'))
    fluid_model.add(LeakyReLU(alpha=0.1))
    fluid_model.add(Dropout(0.45))
    fluid_model.add(Dense(num_classes, activation='softmax'))
    
    fluid_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    fluid_model.summary()
    
    fluid_model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_Y))
    
    fluid_model.save(MODEL_PATH)

# for performance testing
test_Y = np.argmax(our_test_label, axis=1)

print_accuracy(our_test_X, test_Y, label_names, MODEL_PATH)
            
        
        
        
    
    
    
