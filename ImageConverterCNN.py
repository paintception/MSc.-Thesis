import os
import numpy as np
import cv2
import time

def write_pic(data, path, name):
	
	tmp=np.asarray(data, dtype=np.uint8)
	#tmp = cv2.resize(tmp, (8, 8))	#Noise added to get bigger pictures
	cv2.imwrite(path+name+'.png', tmp)

	#print "Image Saved"

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

	X = np.load('//home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/15000Positions.npy')
	y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/15000Positions.npy')

	print len(X)
	print len(y)

	time.sleep(5)

	General_X = []

	for i in X:
		g = i.reshape((12,64))
		tmp = np.zeros(g.shape[1])
		for ind,j in enumerate(g):
			tmp = tmp+j*(ind+1)
		General_X.append(tmp)

	General_X = np.asarray(General_X)

	#np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/64_125000Positions.npy', General_X)
	#np.save('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/64_125000Labels.npy', y)

	path='/home/matthia/Desktop/MSc.-Thesis/CNNImages/SplittedImages/'
	
	do_path(path)

	new_ev = []

	for ind, pos in enumerate(General_X):

		evaluation = y[ind]
		pos=fix_pic(pos)

		if evaluation == "Equal":
			tmp_path=path+"Equal/"
			write_pic(pos, tmp_path, 'position'+str(ind))
			#write_pic(pos, path, 'position'+str(ind))
		
		elif evaluation == "WW":
			#new_ev.append(evaluation)
			tmp_path=path+"WW/"
			write_pic(pos, tmp_path, 'position'+str(ind))
			#write_pic(pos, path, 'position'+str(ind))

		elif evaluation == "BW":
			#new_ev.append(evaluation)
			tmp_path=path+"BW/"
			write_pic(pos, tmp_path, 'position'+str(ind))
			#write_pic(pos, path, 'position'+str(ind))
		
		"""
		if evaluation >= - 1 and evaluation <=1:
			tmp_path=path+"Equal/"
		elif evaluation > 1:
			tmp_path=path+"WW/"
		elif evaluation < 1:
			tmp_path=path+"BW/"
		"""
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
	
