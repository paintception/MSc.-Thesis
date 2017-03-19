import os
import numpy as np
import cv2

from sklearn import preprocessing

def write_pic(data, path, name):
	
	tmp=np.asarray(data, dtype=np.uint8)
	tmp = cv2.resize(tmp, (64, 64))	#Noise added to get bigger pictures
	cv2.imwrite(path+name+'.png', tmp)

	print "Image Saved"

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

	print "Reading the data"

	X = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Positions.npy')
	y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Labels.npy')

	X = preprocessing.scale(X)

	General_X = []

	for i in X:
		g = i.reshape((12,64))
		tmp = np.zeros(g.shape[1])
		for ind,j in enumerate(g):
			tmp = tmp+j*(ind+1)
		General_X.append(tmp)

	General_X = np.asarray(General_X)

	path='/home/matthia/Desktop/MSc.-Thesis/CNNImages/BasicEvaluation/'
	
	do_path(path)

	for ind, pos in enumerate(General_X):
		
		print "Making Images"

		evaluation = y[ind]
		pos=fix_pic(pos)
		
		if evaluation == "Equal":
			tmp_path=path+"Equal/"
		elif evaluation == "WW":
			tmp_path=path+"WW/"
		elif evaluation == "BW":
			tmp_path=path+"BW/"
		
		write_pic(pos, tmp_path, 'position'+str(ind))
		
		"""
		if evaluation >= - 1 and evaluation <=1:
			tmp_path=path+"Equal/"
		elif evaluation > 1:
			tmp_path=path+"WW/"
		elif evaluation < 1:
			tmp_path=path+"BW/"
		"""
		#write_pic(pos, tmp_path, 'position'+str(ind))
		"""
		if evaluation >= - 0.5 and evaluation < 0.5:
			tmp_path=path+"Equal/"
		elif evaluation > 0.5 and evaluation <= 1.5:
			tmp_path=path+"WSB/"
		elif evaluation > 1.5 and evaluation <= 4:
			tmp_path=path+"WWB/"
		elif evaluation >4:
			tmp_path=path+"WW/"
		elif evaluation < -0.5 and evaluation >= - 1.5:
			tmp_path=path+"BSB/"
		elif evaluation < 1.5 and evaluation >= -4:
			tmp_path=path+"BWB/"
		elif evaluation < -4:
			tmp_path=path+"BW/"
		"""
if __name__ == '__main__':
	main()
	
