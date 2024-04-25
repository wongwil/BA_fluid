import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pandas as pd
import sys
import cv2
from data_augmentation import augment_frame

FRAME_FOLDER_PER_LABEL = '../../local/frames/per_label/{}/'
FRAME_FOLDER_PER_VIAL = '../../local/frames/per_vial/{}/'
FRAME_FILE_NAME = '{}_frame_{}.jpg'
LABELS_TO_USE = {'still', 'wave', 'nearDrops', 'smallDrops', 'drops', 'foam'}
random.seed(200)

    
def get_label_names():
    labels = pd.read_csv('../labeling/1g/labels/vial9_labels.csv')
    label_names = list(labels.columns[2:])
    label_names = [i for i in label_names if i in LABELS_TO_USE]
    return label_names

def set_labels(label_names):
    global LABELS_TO_USE
    LABELS_TO_USE = list(label_names)
    
            
            
def get_image(index, vial_name, per_label = False, label_name = None):
    #img = mpimg.imread((FRAME_FOLDER % vial_name) + (FRAME_FILE_NAME % (vial_name, index)))
    #img_gray = img[:,:,0]
    img = None
    if per_label:
        img = cv2.imread(FRAME_FOLDER_PER_LABEL.format(label_name) + FRAME_FILE_NAME.format(vial_name, index))
    else:
        img = cv2.imread(FRAME_FOLDER_PER_VIAL.format(vial_name) + FRAME_FILE_NAME.format(vial_name, index))
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray
    

def normalize_frames(frames):
    frames = np.array(frames)
    frames = frames.reshape(-1, frames.shape[1], frames.shape[2], 1)
    frames = frames.astype('float32')
    frames = frames / 255.0
    
    return frames


def show_image(index):
        img = mpimg.imread(FRAME_FOLDER_PER_LABEL + (FRAME_FILE_NAME % index))
        img_gray = img[:,:,0]
        #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgplot = plt.imshow(img_gray, cmap='gray')
        reshaped_img = img_gray.reshape((-1,img.shape[0],img.shape[1],1))
        print(reshaped_img)
        plt.title(FRAME_FILE_NAME % index) 
        plt.show()

        
def get_frames(path = '../labeling/1g/labels/vial9_labels.csv', with_augmented = False, with_useless = False):
    labels = pd.read_csv(path)
    file_name = path.split('/')[4]
    vial_name = file_name.split('_')[0]
    
    all_frames = []
    
    labels_values = []

    useless_count = 0
    for index, row in labels.iterrows():
        current_frame = get_image(index, vial_name)
        label_value = -1
        if row.useless == 0 or with_useless:
            if row.still == 1 and 'still' in LABELS_TO_USE:
                label_value = 0
                labels_values.append(label_value)
            elif row.wave == 1 and 'wave' in LABELS_TO_USE:
                label_value = 1
                labels_values.append(label_value)
            elif row.nearDrops == 1 and 'nearDrops' in LABELS_TO_USE:
                label_value = 2
                labels_values.append(label_value)
            elif row.smallDrops == 1 and 'smallDrops' in LABELS_TO_USE:
                label_value = 3
                labels_values.append(label_value)
            elif row.drops == 1 and 'drops' in LABELS_TO_USE:
                label_value = 4
                labels_values.append(label_value)
            elif row.foam == 1 and 'foam' in LABELS_TO_USE:
                label_value = 5
                labels_values.append(label_value)
                
            if label_value >= 0:    
                all_frames.append(current_frame)
    
            if label_value > 1 and label_value < 5 and with_augmented:
                for image in augment_frame(current_frame):
                    all_frames.append(image)
                    labels_values.append(label_value)
            
        else:
            useless_count += 1
        
    all_frames = np.array(all_frames)
    all_frames = all_frames.reshape(-1, all_frames.shape[1], all_frames.shape[2], 1)
    all_frames = all_frames.astype('float32')
    all_frames = all_frames / 255.0
    
    return (all_frames, labels_values)

def balance_data(all_frames, label_values, limit_weights = [3, 2, 1, 1, 1, 1]):
    # shuffle data
    all_frames, label_values = shuffle_frames_labels(all_frames, label_values)
   
    unique_labels = list(set(label_values))
    current_count_per_label = dict()
    for i in range(len(unique_labels)):
        current_count_per_label[unique_labels[i]] = 0
        
    ## find min length
    min_length = 10000
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        count = label_values.count(label)
        if count < min_length:
            min_length = count
    
    balanced_all_frames = []
    balanced_labels = []
    length_limits = np.array(limit_weights)*min_length
    
    for i in range(len(label_values)):
        frame = all_frames[i]
        label = label_values[i] # what is the label of the frame

        if current_count_per_label[label] < int(length_limits[label]):
            balanced_all_frames.append(frame)
            balanced_labels.append(label)
            current_count_per_label[label] += 1
    
    balanced_all_frames = np.array(balanced_all_frames)
    balanced_all_frames = balanced_all_frames.reshape(-1, balanced_all_frames.shape[1], balanced_all_frames.shape[2], 1)
    
    return (balanced_all_frames, balanced_labels)

def shuffle_frames_labels(all_frames, label_values):
    assert len(all_frames) == len(label_values)
    zipped = list(zip(all_frames, label_values))
    random.shuffle(zipped)
    shuffled_frames, shuffled_labels = zip(*zipped)
    return shuffled_frames, shuffled_labels

def count_labels(vial_ids):
    current_count_per_label = dict()
    for label_name in LABELS_TO_USE:
        current_count_per_label[label_name] = 0
        
    vial_file = '../labeling/1g/labels/vial{}_labels.csv'
    
    for i in vial_ids:
        path = vial_file.format(i)
        labels_df = pd.read_csv(path)
        for label_name in LABELS_TO_USE:
            count = labels_df[label_name].sum()
            current_count_per_label[label_name] += count
            
    return current_count_per_label