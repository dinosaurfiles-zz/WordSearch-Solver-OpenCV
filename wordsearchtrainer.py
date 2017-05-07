#!/usr/bin/python
import cv2
import sys
import numpy as np

img = cv2.imread('universities.jpg')

# Add Threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = img[y:y+h, x:x+w]

cv2.imshow('Crop', crop)
cv2.waitKey(0)
