import chess.pgn
import time

pgn = open("Games/Lobo, Richard-Cusi, Ronald.pgn")
game = chess.pgn.read_game(pgn)
pgn.close()

node = game
board = chess.Board()

while not node.is_end():
	next_node = node.variation(0)
	node = next_node

	print (node)
	
	print "---------------------"
	time.sleep(0.3)