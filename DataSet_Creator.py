#TODO Add Stockfish move and loop over all the games in pgn

from __future__ import division

import chess
import chess.pgn
import chess.uci
import numpy as np
import pickle

def load_game():
	pgn = open("ExampleGame.pgn")
	game = chess.pgn.read_game(pgn)

	return game

def store_info(board, evaluation):	

	b = np.asarray(board)
	e = np.asarray(evaluation)

	pickle.dump(b, open("Gm_Positions.p", "wb"))
	pickle.dump(e, open("Gm_Evaluations.p", "wb"))
	
def process_game(game):

	engine = chess.uci.popen_engine("/usr/games/stockfish")
	engine.uci()

	GM_board = chess.Board()
	Stock_board = chess.Board()
	node = game
	movetime = 50	#MIlliseconds, the lower the more approximate the value is

	info_handler = chess.uci.InfoHandler()
	engine.info_handlers.append(info_handler)
	
	while not node.is_end():

		engine.position(GM_board)
		b_m = engine.go(movetime=movetime)

		Stock_move = b_m[0]

		info = info_handler.info["score"][1]
		stock_evaluation = info[0]/100 

		next_node = node.variation(0)
		GM_move = str(node.board().san(next_node.move))
		GM_board.push_san(GM_move)

		store_info(GM_board, stock_evaluation)

		node = next_node

def main():
	game = load_game()
	process_game(game)

if __name__ == '__main__':
	main()