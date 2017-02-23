import os 
import numpy as np
import chess

def unpack_data():
	return np.load("Datasets/DS1.dat")

def bitmaps(data):
	
	p_input_vec = []

	for i in data:
		p_input_vec.append(i[0].pieces(chess.PAWN, chess.BLACK))

		print len(p_input_vec)

def main():
	data = unpack_data()
	bitmaps(data)

if __name__ == '__main__':
	main()