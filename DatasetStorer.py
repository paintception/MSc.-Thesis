import numpy as np

def read_data():

	X = []
	y = []
	General_X = []
	ind=1

	with open("/home/matthia/Desktop/MSc.-Thesis/Datasets/Newest.txt", 'r') as f:

		print "Reading the Data"

		for line in f:
			record = line.split(";")
			pieces = [eval(x) for x in record[0:12]]
			piece = [item for sublist in pieces for item in sublist]
			piece = [item for sublist in piece for item in sublist]	

			X.append(piece)
			y.append(float(record[12][:-2]))

	X = np.asarray(X)
	
	new_y = []

	for evaluation in y:
		if evaluation >= - 1 and evaluation <= 1:
			new_y.append("Equal")
		elif evaluation > 1:
			new_y.append("WW")
		elif evaluation < 1:
			new_y.append("BW")

	new_y = np.asarray(new_y)

	np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Positions.npy', X)
	np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Labels.npy', new_y)

def main():
	read_data()

if __name__ == '__main__':
	main()