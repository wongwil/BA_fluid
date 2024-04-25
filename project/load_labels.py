import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import sys
import cv2
from data_augmentation import resize_frames64x64, flip_image, resize_frame64x64, augment_frame

FRAME_FOLDER_PER_LABEL = '../../local/frames/per_label/{}/'
FRAME_FOLDER_PER_VIAL = '../../local/frames/per_vial/{}/'
FRAME_FILE_NAME = '{}_frame_{}.jpg'

def main():
    labels = pd.read_csv('../labeling/1g/labels/vial1_labels.csv')  
    
    ## filter only where drops are true
    #dropFrames = labels.loc[labels['drops'] == True]
    
    #print(len(dropFrames))
    
    all_frames = []
    
    labels_values = []

    for index, row in labels.iterrows():
        # falls man tröpfchen anzeigen muss:
        #reply = input("Press \"Enter\" to show next frame or \"n\" to exit:")
        all_frames.append(get_image(index))
        if row.drops == 0:
            labels_values.append(0)
        else:
            labels_values.append(1)
        
    all_frames = np.array(all_frames)
    all_frames = all_frames.reshape(-1, all_frames.shape[1], all_frames.shape[2], 1)
    all_frames = all_frames.astype('float32')
    all_frames = all_frames / 255.0
    bsp_img = all_frames[470,:,:,:]
    print(all_frames.shape)
    print(all_frames[0,:,:,:])
    
def get_label_names():
    labels = pd.read_csv('../labeling/1g/labels/vial9_labels.csv')
    label_names = list(labels.columns[2:])
    return label_names
            
            
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
    



def show_image(index):
        img = mpimg.imread(FRAME_FOLDER + (FRAME_FILE_NAME % index))
        img_gray = img[:,:,0]
        #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgplot = plt.imshow(img_gray, cmap='gray')
        reshaped_img = img_gray.reshape((-1,img.shape[0],img.shape[1],1))
        print(reshaped_img)
        plt.title(FRAME_FILE_NAME % index) 
        plt.show()
        
def get_frames(path = '../labeling/1g/labels/vial1_labels.csv'):
    labels = pd.read_csv(path)
    file_name = path.split('/')[4]
    vial_name = file_name.split('_')[0]
    
    all_frames = []
    
    labels_values = []

    useless_count = 0
    for index, row in labels.iterrows():
        current_frame = get_image(index, vial_name)
        all_frames.append(current_frame)
        if row.drops == 0:
            labels_values.append(0)
        else:
            labels_values.append(1)
        
    all_frames = np.array(all_frames)
    all_frames = all_frames.reshape(-1, all_frames.shape[1], all_frames.shape[2], 1)
    all_frames = all_frames.astype('float32')
    all_frames = all_frames / 255.0
    
    return (all_frames, labels_values)

def get_augmented_frames(path = '../labeling/1g/labels/vial1_labels.csv'):
    labels = pd.read_csv(path)
    file_name = path.split('/')[4]
    vial_name = file_name.split('_')[0]
    
    all_frames = []
    
    labels_values = []

    for index, row in labels.iterrows():
        current_frame = get_image(index, vial_name)
        all_frames.append(current_frame)
        if row.drops == 0:
            labels_values.append(0)
        else:
            labels_values.append(1)
            for image in augment_frame(current_frame):
                all_frames.append(image)
                labels_values.append(1)
            
        
    all_frames = np.array(all_frames)
    all_frames = all_frames.reshape(-1, all_frames.shape[1], all_frames.shape[2], 1)
    all_frames = all_frames.astype('float32')
    all_frames = all_frames / 255.0
    
    return (all_frames, labels_values)

def get_relevant_frames(path = '../labeling/1g/labels/vial1_labels.csv'):
    labels = pd.read_csv(path)
    file_name = path.split('/')[4]
    vial_name = file_name.split('_')[0]
    
    all_frames = []
    
    labels_values = []

    for index, row in labels.iterrows():
        current_frame = get_image(index, vial_name)
        if row.drops == 1:
            all_frames.append(current_frame)
            labels_values.append(1)
            for image in augment_frame(current_frame):
                all_frames.append(image)
                labels_values.append(1)
            
        
    all_frames = np.array(all_frames)
    all_frames = all_frames.reshape(-1, all_frames.shape[1], all_frames.shape[2], 1)
    all_frames = all_frames.astype('float32')
    all_frames = all_frames / 255.0
    
    return (all_frames, labels_values)
    

if __name__ == "__main__":
    main()