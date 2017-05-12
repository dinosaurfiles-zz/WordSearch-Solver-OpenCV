#!/usr/bin/python
import tkinter as tk
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np

from PIL import ImageTk, Image

class Application(tk.Frame):
	def __init__(self, master):
		super().__init__(master)
		self.pack()
		self.wordSearchMain()
		self.inputList = []
		self.imageFile = "/home/stuxnet/Documents/Major/CMSC197 - Computer Vision/Lab Activities/Final Project/universities.png"
		self.create_widgets()

	def create_widgets(self):
		self.image = tk.PhotoImage(file=self.imageFile)
		self.imgLabel = tk.Label(self, image=self.image)

		# self.selectedPhotoImage = tk.PhotoImage(file=self.imageFile2)
		# self.imgLabel.config(image=self.selectedPhotoImage)

		self.imgLabel.grid(row=1, column=1, rowspan=5, columnspan=4)

		self.filePickerLabel = tk.Label(self, text="Open an image: ").grid(row=1, column=5)
		self.filePicker = tk.Button(self, text="File Picker", command=self.openFile).grid(row=1, column=6)

		self.wordEntryLabel = tk.Label(self, text="Enter word: ").grid(row=2, column=5)
		self.wordEntry = tk.Entry(self)
		self.wordEntry.grid(row=2, column=6, columnspan=2)
		self.wordEntryButton = tk.Button(self, text="Submit", command=self.submitWord).grid(row=2, column=8)

		self.historyLabel = tk.Label(self, text="Input history: ").grid(row=3, column=6, rowspan=2)
		self.history = tk.Label(self, text="\n".join(map(str, self.inputList)))
		self.history.grid(row=3, column=7, rowspan=2, columnspan=2)

	def openFile(self):
		self.selectedImage = askopenfilename(filetypes=(("PNG files", "*.png"), ("All files", "*.*")))
		self.selectedPhotoImage = tk.PhotoImage(file=self.selectedImage)
		self.imgLabel.configure(image=self.selectedPhotoImage)
		self.inputList = []

	def submitWord(self):
		self.newInput = self.wordEntry.get()
		self.inputList.append(self.newInput)
		self.history.configure(text="\n".join(map(str, self.inputList)))
		self.wordSearchAlgo(self.newInput)

	def wordSearchMain(self):
		# Load training data
		samples = np.loadtxt("data/trainingsamples.data", np.float32)
		responses = np.loadtxt("data/trainingresponses.data", np.float32)
		responses = responses.reshape((responses.size, 1))

		# Board
		self.board = []
		self.boardDetail = []

		# Load model
		model = cv2.ml.KNearest_create()
		model.train(samples, cv2.ml.ROW_SAMPLE, responses)

		# Load image
		img = cv2.imread("/home/stuxnet/Documents/Major/CMSC197 - Computer Vision/Lab Activities/Final Project/universities.png")
		out = np.zeros(img.shape, np.uint8)
		imgPlain = img.copy()
		self.imgAnswer = img.copy()

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
					self.board.append(chr(results[0][0]))
					self.boardDetail.append((x, y, w, h))

		# Make self.board using the model
		self.board = list(reversed(self.board))
		self.board = np.array([self.board])
		self.board = np.reshape(self.board, (-1, 17))
		self.boardDetail = list(reversed(self.boardDetail))

	def wordSearchAlgo(self, word):
		orientation = 0
		flag = False
		answer = []
		for x in range(len(self.board)):
			for y in range(len(self.board[x])):

				if self.board[x][y] == word[0]:

					# North
					# Check northbound if length is possible for a word
					if x >= len(word) - 1 and not flag:
						for wIndex in range(len(word)):
							if self.board[x - wIndex][y] == word[wIndex]:
								answer.append((x - wIndex, y))
								flag = True
								orientation = 0
							else:
								answer = []
								flag = False
								break

					# East
					# Check eastbound if length is possible for a word
					if (len(self.board[x]) - y >= len(word)) and not flag:
						for wIndex in range(len(word)):
							if self.board[x][y + wIndex] == word[wIndex]:
								answer.append((x, y + wIndex))
								flag = True
								orientation = 2
							else:
								answer = []
								flag = False
								break

					# South
					# Check southbound if length is possible for a word
					if (len(self.board) - x) >= len(word) and not flag:
						for wIndex in range(len(word)):
							if self.board[x + wIndex][y] == word[wIndex]:
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
							if self.board[x][y - wIndex] == word[wIndex]:
								answer.append((x, y - wIndex))
								flag = True
								orientation = 6
							else:
								answer = []
								flag = False
								break

					# NorthEast
					# Check northeastbound if length is possible for a word
					if y + len(word) <= len(self.board) and x - len(word) + 1 >= 0 and not flag:
						for wIndex in range(len(word)):
							if self.board[x - wIndex][y + wIndex] == word[wIndex]:
								answer.append((x - wIndex, y + wIndex))
								flag = True
								orientation = 1
							else:
								answer = []
								flag = False
								break

					# SouthEast
					# Check southeastbound if length is possible for a word
					if y + len(word) <= len(self.board) and x + len(word) <= len(self.board) and not flag:
						for wIndex in range(len(word)):
							if self.board[x + wIndex][y + wIndex] == word[wIndex]:
								answer.append((x + wIndex, y + wIndex))
								flag = True
								orientation = 3
							else:
								answer = []
								flag = False
								break

					# SouthWest
					# Check southwestbound if length is possible for a word
					if y - len(word) + 1 >= 0 and x + len(word) <= len(self.board) and not flag:
						for wIndex in range(len(word)):
							if self.board[x + wIndex][y - wIndex] == word[wIndex]:
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
							if self.board[x - wIndex][y - wIndex] == word[wIndex]:
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
			firstCoord = self.boardDetail[(answer[0][0] * len(self.board)) + answer[0][1]]
			lastCoord = self.boardDetail[(answer[-1][0] * len(self.board)) + answer[-1][1]]

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
			cv2.polylines(self.imgAnswer,[pts],True,(0, 0, 255), 2)

			

if __name__ == '__main__':
	root = tk.Tk()
	wordSearchApp = Application(master=root)
	wordSearchApp.master.title("Word Search Solver")
	wordSearchApp.master.geometry("1100x700")
	wordSearchApp.master.resizable(width=False, height=False)
	wordSearchApp.mainloop()
