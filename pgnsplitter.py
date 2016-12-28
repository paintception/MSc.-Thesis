import chess.pgn

pgn = open("game.pgn")

first_game = chess.pgn.read_game(pgn)

while first_game: 
    game_name = first_game.headers['White'] + '-' + first_game.headers['Black']
    out = open('Games/'+game_name+'.pgn', 'w')
    print(game_name)
    exporter = chess.pgn.FileExporter(out)
    first_game.accept(exporter)
    first_game = chess.pgn.read_game(pgn)

