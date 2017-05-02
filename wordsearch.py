#!/usr/bin/python
import cv2
import numpy as np

# Load image
img = cv2.imread('test.jpg', 0)

# Display image
cv2.imshow('image', img)

# Get Key
k = cv2.waitKey(0) & 0xFF
if k == 27: #Esc key
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('testgray.jpg', img)
    cv2.destroyAllWindows()
