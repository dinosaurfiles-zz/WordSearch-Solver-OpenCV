#!/usr/bin/python
import sys

import cv2
import numpy as np

# Load training data
samples = np.loadtxt('data/trainingsamples.data',np.float32)
responses = np.loadtxt('data/trainingresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

# Board
board = []

# Load model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Load image
img = cv2.imread('universities.jpg')
out = np.zeros(img.shape, np.uint8)
imgbak = img.copy()

# Grayscale, blur, and thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Find contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Set rectangles
for i in range(len(contours)):
	# Find rectangles that are the size of a character
	if cv2.contourArea(contours[i]) > 50 and cv2.contourArea(contours[i]) < 250:

		# Letters inside the rectangle
		if hierarchy[0][i][3] == -1:
			[x, y, w, h] = cv2.boundingRect(contours[i])
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

			# Find nearest model using KNearest
			roi = thresh[y:y+h, x:x+w]
			roismall = cv2.resize(roi, (10, 10))
			roismall = roismall.reshape((1, 100))
			roismall = np.float32(roismall)
			retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)

			# Find resulting letter and append to output
			# string = chr(results[0][0])
			board.append(chr(results[0][0]))
			# cv2.putText(out, string, (x, y+h), 0, 1, (0, 255, 0))

# Make board using the model
board = list(reversed(board))
board = np.array([board])
board = np.reshape(board, (-1, 17))

print(board)
