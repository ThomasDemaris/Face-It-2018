#!/usr/bin/env python

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

#Set the right value for the filename
n=1
while os.path.isfile(os.path.join('celebrity_faces', f'{n}.jpg')) == True:
                n += 1

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#Capture a frame using spacebar
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image = frame
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

#Crop picture
cropped = image[16:464, 136:504]

#Downsize and save picture
dim = (92, 112)
resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite(os.path.join('celebrity_faces', f'{n}.jpg'), resized)


