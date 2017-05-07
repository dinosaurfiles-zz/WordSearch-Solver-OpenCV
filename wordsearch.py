#!/usr/bin/python
import sys
import cv2
import numpy as np

# Load training data
samples = np.loadtxt('data/trainingsamples.data',np.float32)
responses = np.loadtxt('data/trainingresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

# Load model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Load image
img = cv2.imread('universities.jpg')
imgbak = img.copy()

# Grayscale, blur, and thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Find contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Set rectangles
for cnt in reversed(contours):
	# Find rectangles that are the size of a character
	# TODO: Remove contours that are inside a contour
	if cv2.contourArea(cnt) > 50 and cv2.contourArea(cnt) < 250:
		[x, y, w, h] = cv2.boundingRect(cnt)
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
		# TODO: Find nearest model using KNN

cv2.imshow('Image', img)
cv2.imshow('ImageBak', imgbak)
cv2.waitKey(0)
