#!/usr/bin/python
import tkinter as tk
from PIL import ImageTk, Image

class Application(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.pack()
		self.create_widgets()

	def create_widgets(self):
		self.image = tk.PhotoImage(file="universities.png")
		self.imgLabel = tk.Label(self, image=self.image)
		self.imgLabel.grid(row=1, column=1, rowspan=5, columnspan=4)

		self.filePickerLabel = tk.Label(self, text="Open an image: ").grid(row=1, column=5)
		self.filePicker = tk.Button(self, text="File Picker").grid(row=1, column=6)

		self.wordEntryLabel = tk.Label(self, text="Enter word: ").grid(row=2, column=5)
		self.wordEntry = tk.Entry(self).grid(row=2, column=6, columnspan=2)
		self.wordEntryButton = tk.Button(self, text="Submit").grid(row=2, column=8)

		self.historyLabel = tk.Label(self, text="Input history: ").grid(row=3, column=6, rowspan=2)
		self.history = tk.Label(self, text="washington\noklahoma\nthunders\nlakers").grid(row=3, column=7, rowspan=2, columnspan=2)

root = tk.Tk()
wordSearchApp = Application(master=root)
wordSearchApp.master.title("Word Search Solver")
wordSearchApp.master.geometry("1200x600")
wordSearchApp.master.resizable(width=False, height=False)
wordSearchApp.mainloop()
