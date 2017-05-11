#!/usr/bin/python
import sys
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='WordSearchTrainer')
parser.add_argument('--trainimg', default='training.png')
args = parser.parse_args()

# Load image and make a backup
img = cv2.imread(args.trainimg)
imgbak = img.copy()

# Grayscale, blur, and thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Find contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ready samples and responses
samples =  np.empty((0, 100))
responses = []

# Keys a-z
keys = [i for i in range(97, 123)]

# Enter inputs for all contours
for cnt in reversed(contours):
	if cv2.contourArea(cnt) > 50:
		[x, y, w, h] = cv2.boundingRect(cnt)

		if h > 18:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			roi = thresh[y:y+h, x:x+w]
			roismall = cv2.resize(roi, (10, 10))
			cv2.imshow('Green: indicates already received input. Red: current letter', img)
			key = cv2.waitKey(0)
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

			if key == 27:
				sys.exit()
			elif key in keys:
				# Save responses
				responses.append(key)
				sample = roismall.reshape((1, 100))
				samples = np.append(samples, sample, 0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("Training Complete")

# Save training data
np.savetxt('data/trainingsamples.data', samples)
np.savetxt('data/trainingresponses.data', responses)
