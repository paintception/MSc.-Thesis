import chess
import time

from itertools import islice

def create_moves(board):

	l_moves = []
	moves = board.legal_moves

	for i in moves:
		l_moves.append(i)

	return l_moves

def create_tree(board):

	total_moves = create_moves(board)
	list_boards = []

	new_list_boards = []
	
	for i in range(0, len(total_moves)):
		b = board.copy()
		list_boards.append(board.copy())
	
	for i, j in zip(list_boards, total_moves):
		list_boards.append(i.push(j))

	return [x for x in list_boards if x is not None]
		
def main():

	depth = 3
	starting_board = chess.Board()
	new_list = create_tree(starting_board)
	
	tracker = []

	for i in xrange(0, depth):
		for b in new_list:
			res = create_tree(b)
			tracker.append(res)
		new_list = res

	for position in tracker:
		for moves in position:
			print(moves)
			print "------------------------"
			time.sleep(0.5)

if __name__ == '__main__':
	main()