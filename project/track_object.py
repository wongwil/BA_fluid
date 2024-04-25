# -*- coding: utf-8 -*-
import cv2
import numpy as np

def main():
    videoSrc = '../../local/filmed/2_24_03/GH013735.mp4'
    
    video = cv2.VideoCapture(videoSrc)
    
    _, first_frame = video.read()
    x = 1100
    y = 700
    
    width = 200
    height = 200
    
    roi = first_frame[y: y+height, x: x+width]
    show_frame("ROI", roi)
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    #hsv_roi = cv2.
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    try:
        while True:
            _, frame = video.read()
            

            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
            show_frame("Mask", mask)
            
            _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
            x, y, w, h = track_window
            
            cv2.rectangle(frame, (x,y), (x + w, y+h),  (0, 255, 0), 2)
            show_frame("Video raw", frame)
            key = cv2.waitKey(30)
            
            if key == 27: # esc key to stop
                break
    except Exception as e:
        print(str(e))
        
    video.release()
    cv2.destroyAllWindows()

def show_frame(name, frame):
    try:
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25) 
        cv2.imshow(name, small_frame)
    except Exception as e:
        print("failed resizing")
        

if __name__ == "__main__":
    main()