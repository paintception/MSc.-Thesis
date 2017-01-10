import chess.pgn
import time

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

inp = open('Games/Lobo, Richard-Cusi, Ronald.pgn')

game = chess.pgn.read_game(inp)

node = game 

while node.variations:  #For every node in the pgn extracts bitmap representation for every piece type
    
    next_node = node.variation(0)

    P_input_vec.append(node.board().pieces(chess.PAWN, chess.WHITE))
    R_input_vec.append(node.board().pieces(chess.ROOK, chess.WHITE))
    N_input_vec.append(node.board().pieces(chess.KNIGHT, chess.WHITE))
    B_input_vec.append(node.board().pieces(chess.BISHOP, chess.WHITE))
    Q_input_vec.append(node.board().pieces(chess.QUEEN, chess.WHITE))
    K_input_vec.append(node.board().pieces(chess.KING, chess.WHITE))

    p_input_vec.append(node.board().pieces(chess.PAWN, chess.BLACK))
    r_input_vec.append(node.board().pieces(chess.ROOK, chess.BLACK))
    n_input_vec.append(node.board().pieces(chess.KNIGHT, chess.BLACK))
    b_input_vec.append(node.board().pieces(chess.BISHOP, chess.BLACK))
    q_input_vec.append(node.board().pieces(chess.QUEEN, chess.BLACK))
    k_input_vec.append(node.board().pieces(chess.KING, chess.BLACK))

    time.sleep(0.2)
    
    node = next_node
