import chess.pgn
import json

pgn = open("GameInputs/KingBase2016-03-A00-A39.pgn")

first_game = chess.pgn.read_game(pgn)

while first_game: 
    
    game_info = {'Moves':[] , 'Result':""}
    white = True
    tup = [None,None]
    
    node = first_game 
    while node.variations:
        next_node = node.variation(0)
        move = node.board().san(next_node.move)
        if white:
            tup = [None,None]
            tup[0] = move
        else:
            tup[1] = move
            game_info['Moves'].append(tup)
            
        white = not white
        node = next_node
    
    if not white:
        game_info['Moves'].append(tup)
    
    game_info['Result'] = first_game.headers['Result']
    f = open('Games/'+first_game.headers["White"]+'-'+ first_game.headers["Black"], 'w')
    f.write(json.dumps(game_info))
    f.close()
    first_game = chess.pgn.read_game(pgn)

pgn.close()

