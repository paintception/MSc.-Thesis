import chess.pgn

inp = open('Games/Fischer, Robert J.-Blort, Boris V..pgn')

game = chess.pgn.read_game(inp)

node = game 
while node.variations:
    next_node = node.variation(0)
    
    print(node.board())
    print('\n')
    node = next_node
