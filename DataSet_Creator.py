#TODO Add Stockfish move and loop over all the games in pgn
#Check also movetime

from __future__ import division

import chess
import chess.pgn
import chess.uci
import pandas as pd
import numpy as np

columns = ['WhiteP','WhiteR','WhiteN','WhiteB','WhiteQ','WhiteK',
'BlackP','BlackR','BlackN','BlackB','BlackQ','BlackK','Evaluation']

df = pd.DataFrame([], columns=columns)

def load_game():
	pgn = open("ExampleGame.pgn")
	game = chess.pgn.read_game(pgn)

	return game

def extract_bitmaps(board, e, iter):

	P_input_vec = []
	R_input_vec = []
	N_input_vec = []
	B_input_vec = []
	Q_input_vec = []
	K_input_vec = []

	p_input_vec = []
	r_input_vec = []
	n_input_vec = []
	b_input_vec = []
	q_input_vec = []
	k_input_vec = []

	P_input_vec.append(board.pieces(chess.PAWN, chess.WHITE))
	R_input_vec.append(board.pieces(chess.ROOK, chess.WHITE))
	N_input_vec.append(board.pieces(chess.KNIGHT, chess.WHITE))
	B_input_vec.append(board.pieces(chess.BISHOP, chess.WHITE))
	Q_input_vec.append(board.pieces(chess.QUEEN, chess.WHITE))
	K_input_vec.append(board.pieces(chess.KING, chess.WHITE))

	p_input_vec.append(board.pieces(chess.PAWN, chess.BLACK))
	r_input_vec.append(board.pieces(chess.ROOK, chess.BLACK))
	n_input_vec.append(board.pieces(chess.KNIGHT, chess.BLACK))
	b_input_vec.append(board.pieces(chess.BISHOP, chess.BLACK))
	q_input_vec.append(board.pieces(chess.QUEEN, chess.BLACK))
	k_input_vec.append(board.pieces(chess.KING, chess.BLACK))

	int_P = [int(x) for x in P_input_vec]
	int_R = [int(x) for x in R_input_vec]
	int_N = [int(x) for x in N_input_vec]
	int_B = [int(x) for x in B_input_vec]
	int_Q = [int(x) for x in Q_input_vec]
	int_K = [int(x) for x in K_input_vec]

	int_p = [int(x) for x in p_input_vec]
	int_r = [int(x) for x in r_input_vec]
	int_n = [int(x) for x in n_input_vec]
	int_b = [int(x) for x in b_input_vec]
	int_q = [int(x) for x in q_input_vec]
	int_k = [int(x) for x in k_input_vec]

	column = "WhiteP"
	wp = int_P
	df.at[iter, column] = wp

	column = "WhiteR"
	wr = int_R
	df.at[iter, column] = wr

	column = "WhiteN"
	wn = int_N
	df.at[iter, column] = wn
	
	column = "WhiteB"
	wb = int_B
	df.at[iter, column] = wb
	
	column = "WhiteQ"
	wq = int_Q
	df.at[iter, column] = wq
	
	column = "WhiteK"
	wk = int_K
	df.at[iter, column] = wk
	
	column = "BlackP"
	bp = int_p
	df.at[iter, column] = bp
	
	column = "BlackR"
	br = int_r
	df.at[iter, column] = br

	column = "BlackN"
	bn = int_n
	df.at[iter, column] = bn

	column = "BlackB"
	bb = int_b
	df.at[iter, column] = bb
	
	column = "BlackQ"
	bq = int_q
	df.at[iter, column] = bq

	column = "BlackK"
	bk = int_k
	df.at[iter, column] = bk

	column = "Evaluation"
	wp = e
	df.at[iter, column] = wp

	df.to_csv('Datasets/Data.csv')

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
	
	iter = 0

	while not node.is_end():

		engine.position(GM_board)
		b_m = engine.go(movetime=movetime)

		Stock_move = b_m[0]

		info = info_handler.info["score"][1]
		stock_evaluation = info[0]/100 

		next_node = node.variation(0)
		GM_move = str(node.board().san(next_node.move))
		GM_board.push_san(GM_move)

		pos_dic = extract_bitmaps(GM_board, stock_evaluation, iter)
		iter += 1 
		node = next_node

def main():
	
	game = load_game()
	process_game(game)

if __name__ == '__main__':
	main()