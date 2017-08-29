from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import model_from_json
from keras.optimizers import SGD

from matplotlib import pyplot as plt

import numpy
import os
import time
import chess
import numpy as np
import chess
import random
import seaborn

def start_game():
	return chess.Board()

def splitter(inputStr, black):

	inputStr = format(inputStr, "064b")
	tmp = [inputStr[i:i+8] for i in range(0, len(inputStr), 8)]

	for i in xrange(0, len(tmp)):
		tmp2 = list(tmp[i])
		tmp2 = [int(x) * black for x in tmp2]
		tmp[i] = tmp2

	return tmp

def process_position(board):
		
	P = splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
	R = splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)			
	N = splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
	B = splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
	Q = splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)			
	K = splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

	p = splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
	r = splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
	n = splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
	b = splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
	q = splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
	k = splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

	l = P+R+N+B+Q+K+p+r+n+b+q+k
	BitMappedBoard = [item for sublist in l for item in sublist]

	return BitMappedBoard

def load_model_MLP1(dimension_input):

	model = Sequential()
	model.add(Dense(1048, input_dim=dimension_input, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(500, input_dim=dimension_input, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(3, init='normal', activation='elu'))
	model.add(Activation("softmax"))
	model.load_weights("/home/matthia/Desktop/Bobby/MLP1/Bobbyweights.h5")

	print "Model Loaded"

	return model

def load_model_MLP2(dimension_input):

	model = Sequential()
	model.add(Dense(2048, input_dim=dimension_input, init='normal', activation='elu'))
	model.add(Dropout(0.2))
	model.add(Dense(2048, input_dim=dimension_input, init='normal', activation='elu'))
	model.add(Dropout(0.2))
	model.add(Dense(1050, input_dim=dimension_input, init='normal', activation='elu'))
	model.add(Dropout(0.2))
	model.add(Dense(8, init='normal', activation='elu'))
	#model.add(Activation("softmax"))
	model.load_weights("/home/matthia/Desktop/Bobby/8Classes/Bobbyweights.h5")

 	print "Model Loaded"

 	return model

def load_regressionMLP(dimension_input):

	model = Sequential()
	model.add(Dense(200, input_dim = dimension_input, kernel_initializer='normal', activation = 'elu'))
	model.add(BatchNormalization())
	model.add(Dense(200, input_dim = dimension_input, kernel_initializer='normal', activation = 'elu'))
	model.add(BatchNormalization())
	model.add(Dense(1, kernel_initializer='normal'))
	model.load_weights("")

	print "Model Loaded"

	return model

def analyzer_1(out):

	if np.argmax(out) == 0:
		print "Bobby thinks: It is a DRAW"		

	elif np.argmax(out) == 1:
		print "Bobby thinks: White is WINNING"		

	elif np.argmax(out) == 2:
		print "Bobby thinks: Black is WINNING"		

def analyzer_2(out):

	if np.argmax(out) == 0:
		print "Bobby thinks: Black is SLIGHLTY BETTER"		

	elif np.argmax(out) == 1:
		print "Bobby thinks: Black is DEFINITELY BETTER"		

	elif np.argmax(out) == 2:
		print "Bobby thinks: Black is WINNING"		

	elif np.argmax(out) == 3:
		print "Bobby thinks: Black is DEFINITELY WINNING"		

	elif np.argmax(out) == 4:
		print "Bobby thinks: IT IS A DRAW"		

	elif np.argmax(out) == 5:
		print "Bobby thinks: White is DEFINITELY BETTER"		

	elif np.argmax(out) == 6:
		print "Bobby thinks: White is WINNING"		

	elif np.argmax(out) == 7:
		print "Bobby thinks: White is DEFINITELY WINNING"		


def Bobby(board, model):

	while not board.is_game_over(True):
				
		print(board.pieces(chess.KING, chess.WHITE))
		print(board.pieces(chess.KING, chess.BLACK))
		print "-------------------------"
		print(board)
		print "Make a move"
		
		move = random.choice(list(board.legal_moves))
		board.push(move)
		#board.push_san(move)
		position = process_position(board)
		
		pos = np.expand_dims(position, axis=0)
		out = model.predict(pos)
		print(out)
		print np.argmax(out)
		analyzer_2(out)
		time.sleep(0.2)
		
def main():

	board = start_game()

	print "Select which mode of Bobby you want to test"
	print "1 = MLP1_Simple, 2 = MLP1_Complex, 4 = RegressionMLP"

	mode = raw_input()

	if mode == "1":
		
		dimension_input = 768
		model = load_model_MLP1(dimension_input)
		sgd = "Adam"
		model.compile(optimizer=sgd, loss='categorical_crossentropy')
		out = Bobby(board)
		analyzer_1(out)
		time.sleep(0.2)
		
	elif mode == "2":

		print "Ok"

		dimension_input = 768
		model = load_model_MLP2(dimension_input)
		sgd = SGD(lr=0.1)
		model.compile(optimizer=sgd, loss='categorical_crossentropy')
		Bobby(board, model)
		
	elif mode == "3":
		
		dimension_input = 768
		model = load_regressionMLP(dimension_input)
		sgd = SGD(lr=0.00001)
		model.compile(optimizer=sgd, loss="mean_squared_error")
		out = Bobby(board)
		print(out)		

if __name__ == '__main__':
	main()