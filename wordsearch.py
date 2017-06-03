#!/usr/bin/python
import sys

import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='WordSearch solver using OpenCV')
parser.add_argument('--trainingsamples', default='data/trainingsamples.data')
parser.add_argument('--trainingresponses', default='data/trainingresponses.data')
parser.add_argument('--image', default='universities.png')
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

		if board[x][y] == word[0]:

			# North
			# Check northbound if length is possible for a word
			if x >= len(word) - 1 and not flag:
				for wIndex in range(len(word)):
					if board[x - wIndex][y] == word[wIndex]:
						answer.append((x - wIndex, y))
						flag = True
						orientation = 0
					else:
						answer = []
						flag = False
						break

			# East
			# Check eastbound if length is possible for a word
			if (len(board[x]) - y >= len(word)) and not flag:
				for wIndex in range(len(word)):
					if board[x][y + wIndex] == word[wIndex]:
						answer.append((x, y + wIndex))
						flag = True
						orientation = 2
					else:
						answer = []
						flag = False
						break

			# South
			# Check southbound if length is possible for a word
			if (len(board) - x) >= len(word) and not flag:
				for wIndex in range(len(word)):
					if board[x + wIndex][y] == word[wIndex]:
						answer.append((x + wIndex, y))
						flag = True
						orientation = 4
					else:
						answer = []
						flag = False
						break

			# West
			# Check westbound if length is possible for a word
			if (((y+1) - len(word) ) >= 0) and not flag:
				for wIndex in range(len(word)):
					if board[x][y - wIndex] == word[wIndex]:
						answer.append((x, y - wIndex))
						flag = True
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

			# SouthEast
			# Check southeastbound if length is possible for a word
			if y + len(word) <= len(board) and x + len(word) <= len(board) and not flag:
				for wIndex in range(len(word)):
					if board[x + wIndex][y + wIndex] == word[wIndex]:
						answer.append((x + wIndex, y + wIndex))
						flag = True
						orientation = 3
					else:
						answer = []
						flag = False
						break

			# SouthWest
			# Check southwestbound if length is possible for a word
			if y - len(word) + 1 >= 0 and x + len(word) <= len(board) and not flag:
				for wIndex in range(len(word)):
					if board[x + wIndex][y - wIndex] == word[wIndex]:
						answer.append((x + wIndex, y - wIndex))
						flag = True
						orientation = 5
					else:
						answer = []
						flag = False
						break

			# NorthWest
			# Check northwestbound if length is possible for a word
			if y - len(word) + 1 >= 0 and x - len(word) + 1 >= 0 and not flag:
				for wIndex in range(len(word)):
					if board[x - wIndex][y - wIndex] == word[wIndex]:
						answer.append((x - wIndex, y - wIndex))
						flag = True
						orientation = 7
					else:
						answer = []
						flag = False
						break
		if flag:
			break
	if flag:
		break

if not flag:
	print("Word not found in the puzzle")
else:
	# Draw lines
	firstCoord = boardDetail[(answer[0][0] * len(board)) + answer[0][1]]
	lastCoord = boardDetail[(answer[-1][0] * len(board)) + answer[-1][1]]

	# Reverse lines
	if orientation == 4 or orientation == 6 or orientation == 5 or orientation == 7:
		tempOr = firstCoord
		firstCoord = lastCoord
		lastCoord = tempOr

	if orientation == 0 or orientation == 4:
		pts = np.array([
			[lastCoord[0] - 5, lastCoord[1] - 5],
			[lastCoord[0] + 18, lastCoord[1] - 5],
			[lastCoord[0] + 18, firstCoord[1] + firstCoord[3] + 5],
			[lastCoord[0] - 5, firstCoord[1] + firstCoord[3] + 5]],
		np.int32)
	elif orientation == 2 or orientation == 6:
		pts = np.array([
			[firstCoord[0] - 5, firstCoord[1] - 5],
			[lastCoord[0] + lastCoord[2] + 5, firstCoord[1] - 5],
			[lastCoord[0] + lastCoord[2] + 5, firstCoord[1] + 24],
			[firstCoord[0] - 5, firstCoord[1] + 24]],
		np.int32)
	elif orientation == 1 or orientation == 5:
		pts = np.array([
			[firstCoord[0] - 2, firstCoord[1]],
			[lastCoord[0], lastCoord[1] - 2],
			[lastCoord[0] + 18, lastCoord[1] - 2],
			[lastCoord[0] + 18, lastCoord[1] + 15],
			[firstCoord[0] + 12, firstCoord[1] + 20],
			[firstCoord[0] - 2, firstCoord[1] + 20]],
		np.int32)
	elif orientation == 3 or orientation == 7:
		pts = np.array([
			[firstCoord[0] - 5, firstCoord[1] - 5],
			[firstCoord[0] - 5 + 18, firstCoord[1] - 5],
			[lastCoord[0] + 18, lastCoord[1]],
			[lastCoord[0] + 18, lastCoord[1] + 22],
			[lastCoord[0] - 2, lastCoord[1] + 22],
			[firstCoord[0] - 5, firstCoord[1] + 18]],
		np.int32)

	pts = pts.reshape((-1,1,2))
	cv2.polylines(imgAnswer,[pts],True,(0, 0, 255), 2)

	# Show answer
	cv2.imshow("Original", imgPlain)
	cv2.imshow("Original Contoured", img)
	cv2.imshow("Answer", imgAnswer)
	cv2.waitKey(0)
