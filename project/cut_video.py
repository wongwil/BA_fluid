# -*- coding: utf-8 -*-
import os.path
import cv2
import numpy as np 
from crop_functions import fixImages, Vailcen, cutImage

videoSrc = 'E:/_Studium/BA_HDD/videos/demo/resnet/vial41_softbox.avi'
videoOutput = 'E:/_Studium/BA_HDD/videos/03_06/GH013778_cut.MP4'
start_index = 30
end_index = 1500
def main():
    if os.path.isfile(videoOutput):
        print("outputfile already exists!")
        return
        
    video = cv2.VideoCapture(videoSrc)
    success, first_frame = video.read()
    
    fps, frames = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAMES = int(frames)
    
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoWriter = cv2.VideoWriter(videoOutput, fourcc, fps, (width,height))
    
    count_frames = 0
    all_frames = []
    
    while True:
        success, frame = video.read()
        
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_frames.append(gray)
            count_frames += 1
            print(count_frames)
            if count_frames >= start_index and count_frames <= end_index:
                videoWriter.write(frame)  
        else:
            break

        
    fps, frames_output = videoWriter.get(cv2.CAP_PROP_FPS), videoWriter.get(cv2.CAP_PROP_FRAME_COUNT)
    print("finished ")    
    videoWriter.release()
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
