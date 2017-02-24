import re
import fileinput
import os
import sys

def main():
	
	filedata = open("Engine.pgn")
	
	with open("ParsedEngineGames.pgn", 'w') as f:
		for line in filedata:
			line = re.sub("\[FEN .*", "", line)
			line = re.sub("\[Opening .*", "", line)
			line = re.sub("\[SetUp .*", "", line)
			line = re.sub("\[PlyCount .*", "", line)
			f.write(line)

if __name__ == '__main__':
	main()
