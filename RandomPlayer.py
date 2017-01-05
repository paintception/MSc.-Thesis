import chess
import random

class RandomPlayer():

	def __init__(self, board):

		self.board = board

	def make_board(self):

		return chess.Board()

	def choose_move(self):

		move = random.choice(list(self.board.legal_moves))

		return move

	def update_board(self,move):

		return self.board.push(move)

if __name__ == '__main__':

	p = RandomPlayer(chess.Board())

	while not p.board.is_game_over(True):
		p.update_board(p.choose_move())
		
	print(p.board)
	print(p.board.result(True))		