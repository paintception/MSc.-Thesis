import os 
import random

path = "/home/matthia/Desktop/MSc.-Thesis/SetGames"

def getRandomFile():
	files = os.listdir(path)
	index = random.randrange(0, len(files))
	
	return files[index]

for i in xrange(100):
	f = getRandomFile()
	path_to_file = ("/home/matthia/Desktop/MSc.-Thesis/SetGames/FilteredGames/f")
	FILE = open(path_to_file, "w")