import cv2
import dlib
import numpy as np
name = 0
cap = cv2.VideoCapture('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/s1.mpg_vcd/s1/lwae8n.mpg')
# display the fps and related properties
fps = cap.get(cv2.CAP_PROP_FPS)
totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("video fps:", fps)
print("video total number of frames:", totalNoFrames)