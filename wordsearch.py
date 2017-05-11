#!/usr/bin/python
import sys

import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='WordSearch solver using OpenCV')
parser.add_argument('--trainingsamples', default='data/trainingsamples.data')
parser.add_argument('--trainingresponses', default='data/trainingresponses.data')
parser.add_argument('--image', default='universities.jpg')
parser.add_argument('--word', default='rutger')
args = parser.parse_args()

word = args.word

# Load training data
samples = np.loadtxt(args.trainingsamples, np.float32)
responses = np.loadtxt(args.trainingresponses, np.float32)
responses = responses.reshape((responses.size, 1))

# Board
board = []
boardDetail = []

# Load model
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Load image
img = cv2.imread(args.image)
out = np.zeros(img.shape, np.uint8)
imgPlain = img.copy()
imgAnswer = img.copy()

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
			board.append(chr(results[0][0]))
			boardDetail.append((x, y, w, h))

# Make board using the model
board = list(reversed(board))
board = np.array([board])
board = np.reshape(board, (-1, 17))
boardDetail = list(reversed(boardDetail))

print(board)

# Word Search Algorithm
orientation = 0
flag = False
answer = []
for x in range(len(board)):
	for y in range(len(board[x])):

		# TODO: Find N, NE, E, SE, S, SW, W, NW
		if board[x][y] == word[0]:

			# North
			# Check northbound if length is possible for a word
			if x >= len(word) - 1 and not flag:
				wIndex = 0
				for tempx in range(x, -1, -1):
					if wIndex >= len(word):
						break

					if board[tempx][y] != word[wIndex]:
						flag = False
						answer = []
						break
					else:
						flag = True
						answer.append((tempx, y))
						wIndex = wIndex + 1
						orientation = 0

			# East
			# Check eastbound if length is possible for a word
			if (len(board[x]) - y >= len(word)) and not flag:
				for tempy in range(len(word)):
					if board[x][y + tempy] == word[tempy]:
						answer.append((x, y + tempy))
						flag = True
						orientation = 2
					else:
						answer = []
						flag = False
						break

			# South
			# Check southbound if length is possible for a word
			if (len(board) - x) >= len(word) and not flag:
				wIndex = 0
				for tempx in range(x, x+len(word)):
					if board[tempx][y] != word[wIndex]:
						flag = False
						answer = []
						break
					else:
						flag = True
						answer.append((tempx, y))
						wIndex = wIndex + 1
						orientation = 4

			# West
			# Check westbound if length is possible for a word
			if (((y+1) - len(word) ) >= 0) and not flag:
				wIndex = 0
				for tempy in range(y, y - (len(word)), -1):

					if board[x][tempy] == word[wIndex]:
						answer.append((x, tempy))
						flag = True
						wIndex = wIndex + 1
						orientation = 6
					else:
						answer = []
						flag = False
						break

			# NorthEast
			# Check northeastbound if length is possible for a word
			if y + len(word) <= len(board) and x - len(word) + 1 >= 0 and not flag:
				for wIndex in range(len(word)):
					if board[x - wIndex][y + wIndex] == word[wIndex]:
						answer.append((x - wIndex, y + wIndex))
						flag = True
						orientation = 1
					else:
						answer = []
						flag = False
						break


		if flag:
			break
	if flag:
		break

print(answer)

# Draw lines
# if not flag:
# 	print("Word not found in the puzzle")
# else:
# 	firstCoord = boardDetail[(answer[0][0] * len(board)) + answer[0][1]]
# 	lastCoord = boardDetail[(answer[-1][0] * len(board)) + answer[-1][1]]
# 	print("%s %s" % (firstCoord, lastCoord))
#
# 	if orientation == 4 or orientation == 6:
# 		tempOr = firstCoord
# 		firstCoord = lastCoord
# 		lastCoord = tempOr
#
# 	if orientation == 0 or orientation == 4:
# 		pts = np.array([
# 			[lastCoord[0], lastCoord[1]],
# 			[lastCoord[0] + lastCoord[2], lastCoord[1]],
# 			[firstCoord[0] + firstCoord[3], firstCoord[1] + firstCoord[2]],
# 			[firstCoord[0], firstCoord[1] + firstCoord[3]]],
# 		np.int32)
# 	elif orientation == 2 or orientation == 6:
# 		pts = np.array([
# 			[firstCoord[0], firstCoord[1]],
# 			[lastCoord[0] + lastCoord[2], lastCoord[1]],
# 			[lastCoord[0] + lastCoord[2], lastCoord[1] + lastCoord[3]],
# 			[firstCoord[0], firstCoord[1] + firstCoord[3]]],
# 		np.int32)
#
# 	pts = pts.reshape((-1,1,2))
# 	cv2.polylines(imgAnswer,[pts],True,(0, 0, 255), 2)
#
# 	# cv2.imshow("Original Image", img)
# 	# Show answer
# 	cv2.imshow("Answer", imgAnswer)
# 	cv2.waitKey(0)
