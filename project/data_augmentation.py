import numpy as np
import pandas as pd
import cv2

def augment_frame(frame):
    augmented_data = []
    
    flipped_images = flip_image(frame)
    augmented_data.extend(flipped_images)
    augmented_data.append(blur_image(frame))

    #blur flipped images
    for flipped_image in flipped_images:
         augmented_data.append(blur_image(flipped_image))
    
    return augmented_data
    

def flip_image(image):
    
    fliped_images = []
    
    fliped_images.append(np.flipud(image))
    fliped_images.append(np.fliplr(image))
    
    return fliped_images

def blur_image(image):
    blured = cv2.blur(image, (3,3))
    
    return blured

def resize_frame64x64(frame):
    
    resized = cv2.resize(frame, (64,64))
    
    return resized

def augment_data(frames, labels):
    augmented_frames = []
    augmented_labels = []
    
    for i in range(len(frames)):
        current_frame = frames[i]
        label_value = labels[i]
        augmented_frames.append(current_frame)
        augmented_labels.append(label_value)
        for image in augment_frame(current_frame):
            augmented_frames.append(image.reshape(image.shape[0], image.shape[1], 1))
            augmented_labels.append(label_value)
    
    augmented_frames = np.array(augmented_frames)
    augmented_frames = augmented_frames.reshape(-1, augmented_frames.shape[1], augmented_frames.shape[2], 1)
    
    return (augmented_frames, augmented_labels)