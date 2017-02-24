from __future__ import division

import chess
import chess.pgn
import chess.uci
import pandas as pd
import numpy as np
import os

thefile = open('/home/matthia/Desktop/MSc.-Thesis/Datasets/A8099/A8099.txt', 'w')
directory = '/home/matthia/Desktop/MSc.-Thesis/ParsedGames/A8099'

def load_game():

	for root, dirs, filenames in os.walk(directory):
		for f in filenames:
			pgn = open(os.path.join(root, f), 'r')
			game = chess.pgn.read_game(pgn)
			process_game(game)

def splitter(inputStr, black):
	
	inputStr = format(inputStr, "064b")
	tmp = [inputStr[i:i+8] for i in range(0, len(inputStr), 8)]
	for i in xrange(0, len(tmp)):
		tmp2 = list(tmp[i])
		tmp2 = [int(x) * black for x in tmp2]
		tmp[i] = tmp2

	return tmp

def extract_bitmaps(board, e):
	
	P_input = splitter(int(board.pieces(chess.PAWN, chess.WHITE)), 1)
	R_input = splitter(int(board.pieces(chess.ROOK, chess.WHITE)), 1)
	N_input = splitter(int(board.pieces(chess.KNIGHT, chess.WHITE)), 1)
	B_input = splitter(int(board.pieces(chess.BISHOP, chess.WHITE)), 1)
	Q_input = splitter(int(board.pieces(chess.QUEEN, chess.WHITE)), 1)
	K_input = splitter(int(board.pieces(chess.KING, chess.WHITE)), 1)

	p_input = splitter(int(board.pieces(chess.PAWN, chess.BLACK)), -1)
	r_input = splitter(int(board.pieces(chess.ROOK, chess.BLACK)), -1)
	n_input = splitter(int(board.pieces(chess.KNIGHT, chess.BLACK)), -1)
	b_input = splitter(int(board.pieces(chess.BISHOP, chess.BLACK)), -1)
	q_input = splitter(int(board.pieces(chess.QUEEN, chess.BLACK)), -1)
	k_input = splitter(int(board.pieces(chess.KING, chess.BLACK)), -1)

	thefile.write("%s;" % P_input)
	thefile.write("%s;" % R_input)
	thefile.write("%s;" % N_input)
	thefile.write("%s;" % B_input)
	thefile.write("%s;" % Q_input)
	thefile.write("%s;" % K_input)
	thefile.write("%s;" % p_input)
	thefile.write("%s;" % r_input)
	thefile.write("%s;" % n_input)
	thefile.write("%s;" % b_input)
	thefile.write("%s;" % q_input)
	thefile.write("%s;" % k_input)
	
	thefile.write("%s\n" % e)

def process_game(game):

	positions = []
	evaluations = []

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

		info = info_handler.info["score"][1]
		
		next_node = node.variation(0)
		
		if info[0] is not None:
			
			stock_evaluation = info[0]/100 
			
			GM_move = str(node.board().san(next_node.move))
			GM_board.push_san(GM_move)
			extract_bitmaps(GM_board, stock_evaluation)
		
		node = next_node

if __name__ == '__main__':	
	load_game()
