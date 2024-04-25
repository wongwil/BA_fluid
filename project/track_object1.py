# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path

TOTAL_FRAMES = 0
def main():
    videoSrc = 'E:/_Studium/BA_HDD/videos/16_04/GH013766.MP4'
    videoOutput = 'E:/_Studium/BA_HDD/videos/03_06/GH013766_new_tracked.avi'
    # output has to be avi
    
    if os.path.isfile(videoOutput):
        print("outputfile already exists!")
        return
        
    video = cv2.VideoCapture(videoSrc)
    

    success, first_frame = video.read()
    first_frame = resize_frame(first_frame)
    
    # select overall object
    shaker_bbox = cv2.selectROI("select overall cut", first_frame)
    shaker_old_x = int(shaker_bbox[0])
    shaker_old_y = int(shaker_bbox[1])
    shaker_w = int(shaker_bbox[2])
    shaker_h = int(shaker_bbox[3])
    
    fps, frames = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
    TOTAL_FRAMES = int(frames)
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(videoOutput, fourcc, fps, (shaker_w,shaker_h))
    
    # select track object
    
    # MOSSE is fast
    tracker = cv2.TrackerCSRT().create() 
    #tracker = cv2.TrackerCSRT_create()
    
    tracker_bbox = cv2.selectROI("select track object", first_frame, False)
    tracker.init(first_frame, tracker_bbox)
    tracker_old_x = int(tracker_bbox[0])
    tracker_old_y = int(tracker_bbox[1])
    
    
    cap_read_failed = 0
    count_frames = 0
    while count_frames < 1320:
        try:
            timer = cv2.getTickCount()
            success, frame = video.read()
            
            if success:
                cap_read_failed = 0
                frame = resize_frame(frame) # have to resize the read
                frame_copy = frame.copy()
                success, tracker_bbox = tracker.update(frame)
                
                if success:
                    draw_box(frame, tracker_bbox)
                    # calculate movement from old coordinates
                    tracker_new_x = int(tracker_bbox[0])
                    tracker_new_y = int(tracker_bbox[1])
                    
                    diff_x = tracker_new_x - tracker_old_x 
                    diff_y = tracker_new_y - tracker_old_y
                    
                    print("----------------")
                    print("frame: " +str(count_frames)+ " von "+ str(TOTAL_FRAMES))
                    print("movement x:" + str(diff_x))
                    print("movement y:" + str(diff_y))
                    
                    # update tracker coordinates
                    tracker_old_x = tracker_new_x
                    tracker_old_y = tracker_new_y
                    
                    # update shaker coordiantes
                    shaker_old_x += diff_x
                    shaker_old_y += diff_y
                    draw_box(frame, (shaker_old_x, shaker_old_y, shaker_w, shaker_h))
                    # TODO: add shaker frame to video

                    draw_box(frame_copy, tracker_bbox)
                    crop_img = frame_copy[shaker_old_y:shaker_old_y+shaker_h, shaker_old_x:shaker_old_x+shaker_w]
                    videoWriter.write(crop_img)
                    count_frames += 1
                    # cv2.imshow('Stabilized',crop_img)
                else:
                    print("lost tracker")
                    draw_text(frame, "Lost tracker", 60, 60)
                            
                
                fps = cv2.getTickFrequency()/(cv2.getTickCount() - timer)
                draw_text(frame, str(int(fps)), 60, 50)
                
                # cv2.imshow("Raw frames", frame)
                key = cv2.waitKey(1)
                
                if key == 27: # esc key to stop
                    break
            else:
                cap_read_failed += 1
                if cap_read_failed > 300:
                    break
                print("could not read frame from video")
      
        except Exception as e:
            print("exception: " + str(e))
            break
            


    fps, frames_output = videoWriter.get(cv2.CAP_PROP_FPS), videoWriter.get(cv2.CAP_PROP_FRAME_COUNT)
    print("finished ")    
    videoWriter.release()
    video.release()
    cv2.destroyAllWindows()


def resize_frame(frame):
    try:
        return cv2.resize(frame, None, fx=0.9, fy=0.9) 
    except Exception as e:
        print("failed resizing " + str(e))

def draw_box(frame, bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(frame, (x,y), (x + w, y+h),  (0, 255, 0), 2)
    
    
def draw_text(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (213,239,255), 2)

if __name__ == "__main__":
    main()