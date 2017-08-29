from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

import time
import chess
import random

import numpy as np 

class GameHandler(object):

	def __init__(self):
		self.StartingPositionsPath = '/home/matthia/Desktop/PositionsSet.txt'
		self.MlpClassificationWeights = '/home/matthia/Desktop/Bobby/8Classes/Bobbyweights.h5'
		self.MlpDimension = 768
		self.NumberSimulationGames = 2
		self.MlpWins = 0
		self.CnnWins = 0
		self.Draws = 0

	def evaluatePositionMlp(self, board, model, tmp_move, boardToPlay):

		optimalMoves = []

		pos = np.expand_dims(board, axis=0)
		out = model.predict(pos)

		#print np.argmax(out)
		return np.argmax(out)

	def loadMlpClassificationModel(self):

		model = Sequential()
		model.add(Dense(2048, input_dim=self.MlpDimension, init='normal', activation='elu'))
		model.add(Dropout(0.2))
		model.add(Dense(2048, input_dim=self.MlpDimension, init='normal', activation='elu'))
		model.add(Dropout(0.2))
		model.add(Dense(1050, input_dim=self.MlpDimension, init='normal', activation='elu'))
		model.add(Dropout(0.2))
		model.add(Dense(8, init='normal', activation='elu'))
		#model.add(Activation("softmax"))
		model.load_weights("/home/matthia/Desktop/Bobby/8Classes/Bobbyweights.h5")

 		return model

 	def splitter(self, inputStr, black):

 		inputStr = format(inputStr, "064b")
		tmp = [inputStr[i:i+8] for i in range(0, len(inputStr), 8)]

		for i in xrange(0, len(tmp)):
			tmp2 = list(tmp[i])
			tmp2 = [int(x) * black for x in tmp2]
			tmp[i] = tmp2

		return tmp

 	def shapeBoardMlp(self, board):

		P = self.splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
		R = self.splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)			
		N = self.splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
		B = self.splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
		Q = self.splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)			
		K = self.splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

		p = self.splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
		r = self.splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
		n = self.splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
		b = self.splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
		q = self.splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
		k = self.splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

		l = P+R+N+B+Q+K+p+r+n+b+q+k

		BitMappedBoard = [item for sublist in l for item in sublist]

		return BitMappedBoard

 	def loadCnnModel(self):
 		pass

	def loadStartingPositions(self, position):
		return chess.Board(fen=position)
	
	def chooseWhite(self):
		return random.randint(0,1)

	def createSetMoves(self, board):
		return board.legal_moves

	def updateGameStatsWhite(self, result):

		if result == '1-0':
			self.MlpWins = self.MlpWins + 1
		elif result == '0-1':
			self.CnnWins = self.CnnWins + 1
		elif result == '1/2-1/2':
			self.Draws = self.Draws + 1
		else:
			pass

	def updateGameStatsBlack(self, result):

		if result == '1-0':
			self.CnnWins = self.CnnWins + 1
		elif result == '0-1':
			self.MlpWins = self.MlpWins + 1
		elif result == '1/2-1/2':
			self.Draws = self.Draws + 1
		else:
			pass

	def makeAllMovesMlp(self, boardToPlay, setMoves, MlpModel):
		
		optimalOutput = 0
		candidateMoves = []

		while len(setMoves) != 0:
			tmp_move = random.choice(setMoves)
			tmpBoard = deepcopy(boardToPlay)

			tmpBoard.push(tmp_move)
			shapedBoardMlp = self.shapeBoardMlp(tmpBoard)
			out = self.evaluatePositionMlp(shapedBoardMlp, MlpModel, tmp_move, tmpBoard)
			setMoves.remove(tmp_move)

			if out > optimalOutput:
				candidateMoves = [] 
				optimalOutput = out
				candidateMoves.append(tmp_move)
			
			elif out == optimalOutput:
				candidateMoves.append(tmp_move)

			#print "Output to beat: ", optimalOutput
			#print "Candidate Moves: ", candidateMoves

		return candidateMoves

	def startGame(self, boardToPlay, MlpModel):

		#White = self.chooseWhite()

		White = 0

		if White == 0:	#Mlp is White
			
			WhitePlayer = MlpModel

			while not boardToPlay.is_game_over(claim_draw=True):

				setMoves = list(self.createSetMoves(boardToPlay))
				candidateSetMoves = list(self.makeAllMovesMlp(boardToPlay, setMoves, MlpModel))

				move = random.choice(candidateSetMoves)
				boardToPlay.push(move)

				print(boardToPlay)
				print "-------------------------------"
				time.sleep(0.4)

			result = boardToPlay.result()
			self.updateGameStatsWhite(result)

		"""
		elif White == 1: #CNN is White
			
			while not boardToPlay.is_game_over(claim_draw=True):

				setMoves = self.createSetMoves(boardToPlay)
				#move = random.choice(list(setMoves))

				for move in setMoves:
					self.makeMove() 

			result = boardToPlay.result()
			self.updateGameStatsBlack(result)
		"""

	def main(self):
		
		MlpModel = self.loadMlpClassificationModel()

		with open(self.StartingPositionsPath) as f:
			individualPositions = f.readlines()
		
		for position in individualPositions:
			boardToPlay = self.loadStartingPositions(position)
			for i in xrange(0, self.NumberSimulationGames):
				copiedBoard = deepcopy(boardToPlay)
				
				self.startGame(copiedBoard, MlpModel)

		print "Amount of Draws: ", Gamer.Draws
		print "Amount of MLP Wins: ", Gamer.MlpWins
		print "Amount of CNN Wins: ", Gamer.CnnWins

Gamer = GameHandler()
Gamer.main()
