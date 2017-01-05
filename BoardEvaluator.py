from RandomPlayer import RandomPlayer
import chess.uci

class BoardEvaluator(chess.uci.InfoHandler):

	def __init__(self, player):

		self.player = player
		self.engine = chess.uci.popen_engine("/usr/games/stockfish")
		self.engine.uci()
		
	def update_UCI(self):
		move = self.player.choose_move()
		self.player.update_board(move)
		self.engine.position(self.player.board)
		print("Move", move)
		print("Best Response", self.engine.go(movetime=2000).bestmove)	

p = RandomPlayer(chess.Board())
b = BoardEvaluator(p)

while not p.board.is_game_over(True):
	b.update_UCI()


	