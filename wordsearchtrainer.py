#!/usr/bin/python
import cv2
import sys
import numpy as np

img = cv2.imread('training.jpg')
imgbak = img.copy()

# Add Threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0, 100))
responses = []
keys = [i for i in range(97, 122)]

for cnt in reversed(contours):
	if cv2.contourArea(cnt)>50:
		[x, y, w, h] = cv2.boundingRect(cnt)
		# print("%d %d %d %d" % (x, y, w, h))

		if  h>18:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
			roi = thresh[y:y+h, x:x+w]
			roismall = cv2.resize(roi, (10, 10))
			cv2.imshow('norm', img)
			key = cv2.waitKey(0)
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

			if key == 27:
				sys.exit()
			elif key in keys:
				responses.append(key)
				sample = roismall.reshape((1, 100))
				samples = np.append(samples, sample, 0)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt('data/trainingsamples.data', samples)
np.savetxt('data/trainingresponses.data', responses)
