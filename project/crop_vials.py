# -*- coding: utf-8 -*-
import os.path
import cv2
import numpy as np 
from crop_functions import fixImages, Vailcen, cutImage

videoSrc = 'E:/_Studium/BA_HDD/videos/GH013738_0045_0130_locked2.avi'
videoOutput = '../../local/videos/16_04/test_run.avi'

def main():
    # if os.path.isfile(videoOutput):
    #     print("outputfile already exists!")
    #     return
        
    video = cv2.VideoCapture(videoSrc)
    success, first_frame = video.read()
    
    fps, frames = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAMES = int(frames)
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    
    # TODO: detect height of vial 
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoWriter = cv2.VideoWriter(videoOutput, fourcc, fps, (width,height))
    
    count_frames = 0
    all_frames = []
    template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
    
    while True:
        success, frame = video.read()
        
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_frames.append(gray)
            count_frames += 1
            print("Reading frame " + str(count_frames) + "/" + str(TOTAL_FRAMES))
            # rect_pt, anzvails = Vailcen(gray)
            
            # cv2.imshow('frame',frame)
            # videoWriter.write(frame)  
            # count_frames += 1
            # key = cv2.waitKey(1)

            # if key == 27: # esc key to stop
            #     break
        else:
            break

    #frame_array, max_loc0 = fixImages(all_frames, template)
    rect_pt, anzvails  = Vailcen(all_frames[0])
    img_res_0 = cutImage(all_frames, 0, rect_pt)
    for f in range(len(all_frames)):
        cv2.imshow('test', all_frames[f])
        cv2.waitKey(0)
        
    fps, frames_output = videoWriter.get(cv2.CAP_PROP_FPS), videoWriter.get(cv2.CAP_PROP_FRAME_COUNT)
    print("finished ")    
    videoWriter.release()
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()