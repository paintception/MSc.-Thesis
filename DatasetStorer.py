import numpy as np
import time

def read_data():

	X = []
	y = []
	General_X = []
	ind=1

	with open("/home/matthia/Desktop/MSc.-Thesis/Datasets/C2059.txt", 'r') as f:

		print "Reading the Data"

		for line in f:
			record = line.split(";")
			pieces = [eval(x) for x in record[0:12]]
			piece = [item for sublist in pieces for item in sublist]
			piece = [item for sublist in piece for item in sublist]	

			X.append(piece)
			y.append(float(record[12][:-2]))

	X = np.asarray(X)
	
	print "Flipping the Data"

	odd_numbers = [k for a,k in enumerate(y) if a%2 != 0]
	even_numbers = [k for a,k in enumerate(y) if a%2 == 0]
	flipped_odd = [-a for a in odd_numbers]

	y = []

	for i,j in zip(even_numbers, flipped_odd):
		y.append(i)
		y.append(j)


	if len(X) > len(y):
		X = X.pop()

	equal_cnt = 0
	WW_cnt = 0
	BW_cnt = 0	

	new_y = []
	Pos_X = []

	for pos, evaluation in zip(X,y):
		if evaluation >= - 1.5 and evaluation <= 1.5 and equal_cnt <= 25000:
			equal_cnt += 1
			Pos_X.append(pos)
			new_y.append("Equal")
			print "Equal Position Created"
	
		elif evaluation > 1.5 and WW_cnt <= 25000:
			WW_cnt += 1
			Pos_X.append(pos)
			new_y.append("WW")
			print "WW Position Created"
		
		elif evaluation < 1.5 and BW_cnt <= 25000:
			BW_cnt += 1
			Pos_X.append(pos)
			new_y.append("BW")
			print "BW Position Created"

	new_y = np.asarray(new_y)

	print "Equal Positions: ", equal_cnt
	print "WW Positions: ", WW_cnt
	print "BW Positions: ", BW_cnt

	np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/15000Positions.npy', Pos_X)
	np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/15000Labels.npy', new_y)

	print len(Pos_X)
	print len(new_y)

def main():
	read_data()

if __name__ == '__main__':
	main()
