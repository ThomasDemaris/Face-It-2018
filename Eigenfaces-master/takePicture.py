#!/usr/bin/env python

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

#Set the right value for the filename
n=1
while os.path.isfile(os.path.join('celebrity_faces',f'{n}.jpg')) == True:
                n += 1

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(os.path.join('celebrity_faces',f'{n}.jpg'),frame)
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()