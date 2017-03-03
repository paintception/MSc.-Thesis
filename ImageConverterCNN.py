#Remember to maked the dirs manually

import os
import numpy as np
import cv2

def write_pic(data, path, name):
	
	tmp=np.asarray(data, dtype=np.uint8)
	cv2.imwrite(path+name+'.jpg', tmp)

def do_path(path):
	try:
		os.mkdir(path)
	except:
		pass

def fix_pic(data):
	
	data=data.reshape((8,8))
	ind=data[:]!=0
	data[ind]=data[ind]+13
	data=data/19.*254

	return data

def main():


	X = []
	y = []
	General_X = []
	ind=1

	with open("FicsLabelled.txt", 'r') as f:
		for line in f:
			record = line.split(";")
			pieces = [eval(x) for x in record[0:12]]
			piece = [item for sublist in pieces for item in sublist]
			piece = [item for sublist in piece for item in sublist]	

			X.append(piece)
			y.append(float(record[12][:-2]))

	X = np.asarray(X)

	for i in X:
		g = i.reshape((12,64))
		tmp = np.zeros(g.shape[1])
		for ind,j in enumerate(g):
			tmp = tmp+j*(ind+1)
		General_X.append(tmp)

	General_X = np.asarray(General_X)

	path='/Users/matthiasabatelli/Desktop/MSc.-Thesis/CNNImages/'
	
	do_path(path)

	for ind, pos in enumerate(General_X[:2000]):
		evalation = y[ind]
		pos=fix_pic(pos)
		
		if evalation >= - 0.5 and evalation < 0.5:
			tmp_path=path+"Equal/"
		elif evalation > 0.5 and evalation <= 1.5:
			tmp_path=path+"WSB/"
		elif evalation > 1.5 and evalation <= 4:
			tmp_path=path+"WWB/"
		elif evalation >4:
			tmp_path=path+"WW/"
		elif evalation < -0.5 and evalation >= - 1.5:
			tmp_path=path+"BSB/"
		elif evalation < 1.5 and evalation >= -4:
			tmp_path=path+"BWB/"
		elif evalation < -4:
			tmp_path=path+"BW/"

		write_pic(pos, tmp_path, 'pic'+str(ind))

if __name__ == '__main__':
	main()
	