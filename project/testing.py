# -*- coding: utf-8 -*-
from load_labels import get_frames, get_augmented_frames, get_label_names
from data_augmentation import flip_data, resize_frames64x64
from predict_instability import predict_instability, load_model
import matplotlib.pyplot as plt

MODEL_PATH = 'my_model_vial1_vial3_with_blur.h5'

(our_x, our_y) = get_frames(path = '../labeling/1g/labels/vial9_labels.csv')

label_names = get_label_names()


# testing if the network does good predictions
index = 0
count_ones = 0
count_zeros = 0
count_correct_ones = 0

count_predicted_zeros = 0
count_correct_zeros = 0
load_model(MODEL_PATH)
for frame in our_x:
    frame = frame.reshape(-1, frame.shape[0], frame.shape[1], 1)
    label = our_y[index]
    
    if label == 1:
        (prediction, predictions) = predict_instability(frame, MODEL_PATH)
        count_ones += 1
        print(prediction)
        print(predictions[0])
        if prediction == label:
            #imgplot = plt.imshow(frame[0], cmap='gray')
            #reply = input("Press \"Enter\" to show next frame:")
            count_correct_ones += 1
    elif label == 0 and count_zeros%16 == 0:
        (prediction, predictions) = predict_instability(frame, MODEL_PATH)
        count_predicted_zeros += 1
        count_zeros += 1
        print(prediction)
        print(predictions[0])
        if prediction == label:
            #imgplot = plt.imshow(frame[0], cmap='gray')
            #reply = input("Press \"Enter\" to show next frame:")
            count_correct_zeros += 1
         
    else:
         count_zeros += 1
         
    index += 1
    
accuracy_ones = 1.0 * count_correct_ones / count_ones
accuracy_zeros = 1.0 * count_correct_zeros / count_predicted_zeros
global_accuracy = 1.0 * (count_correct_ones + count_correct_zeros) / (count_ones + count_predicted_zeros)