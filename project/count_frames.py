# -*- coding: utf-8 -*-
import cv2

#src_file = "../../local/1g_rename.avi"
src_file = "../../local/videos/16_04/vial30.mkv"

cap = cv2.VideoCapture(src_file)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame length: " + str(int(length)))

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("height " + str(int(height)))
print("width " + str(int(width)))