# -*- coding: utf-8 -*-
# src: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import cv2
import pandas as pd

#change to new video Source
videoSrc = '../../local/videos/16_04/vial41.mkv'

dest_folder_label = '../../local/frames/per_label/{}/'
dest_folder_vial = '../../local/frames/per_vial/{}/'
imageName ='{}_frame_{}'
labels_path = '../labeling/1g/labels/{}_labels.csv'
vidcap = cv2.VideoCapture(videoSrc)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame length: " + str(int(length)))

vial_name = videoSrc.split('/')[5].split('.')[0]
dest_folder_vial = dest_folder_vial.format(vial_name)
labels_path = labels_path.format(vial_name)

height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

labels = pd.read_csv(labels_path)

def save_per_label(count):
    row = labels.iloc[count]
    counter = 0
    for label in row:
        if label == 1:
            label_name = row.index[counter]
            label_folder = dest_folder_label.format(label_name)
            cv2.imwrite(label_folder+imageName.format(vial_name,count)+'.jpg',image)
        counter += 1
    
    

success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(dest_folder_vial+imageName.format(vial_name,count)+'.jpg',image)     # save frame as JPEG file      
  save_per_label(count)
  success,image = vidcap.read()
  print(str(count)+'/' + str(length) + ' h:' +str(int(height)) + ' w:'+str(int(width)), success)
  count += 1
  
  
