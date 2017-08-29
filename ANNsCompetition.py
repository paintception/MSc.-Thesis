from copy import deepcopy

import chess
import random

class GameHandler(object):

	def __init__(self):
		self.StartingPositionsPath = '/home/matthia/Desktop/PositionsSet.txt'
		self.NumberSimulationGames = 200
		self.MlpWins = 0
		self.CnnWins = 0
		self.Draws = 0

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

	def startGame(self, boardToPlay):

		White = self.chooseWhite()

		if White == 0:	#Mlp is White
			
			while not boardToPlay.is_game_over(claim_draw=True):

				setMoves = self.createSetMoves(boardToPlay)
				move = random.choice(list(setMoves))

				boardToPlay.push(move)

			result = boardToPlay.result()
			self.updateGameStatsWhite(result)

		elif White == 1: #CNN is White
			
			while not boardToPlay.is_game_over(claim_draw=True):

				setMoves = self.createSetMoves(boardToPlay)
				move = random.choice(list(setMoves))

				boardToPlay.push(move)

			result = boardToPlay.result()
			self.updateGameStatsBlack(result)

	def main(self):
		
		with open(self.StartingPositionsPath) as f:
			individualPositions = f.readlines()
		
		for position in individualPositions:
			boardToPlay = self.loadStartingPositions(position)
			for i in xrange(0, self.NumberSimulationGames):
				copiedBoard = deepcopy(boardToPlay)
				
				self.startGame(copiedBoard)

		print "Amount of Draws: ", Gamer.Draws
		print "Amount of MLP Wins: ", Gamer.MlpWins
		print "Amount of CNN Wins: ", Gamer.CnnWins

Gamer = GameHandler()
Gamer.main()