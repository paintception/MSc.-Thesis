import numpy as np 
from matplotlib import pyplot as plt

def makeImage(positions):
    
    i = 0
    
    for pos in positions:
        pos = pos.reshape(8,8)
        fig = plt.matshow(pos)
        plt.axis('off')
        i = i + 1
        plt.savefig('/home/matthia/Desktop/MSc.-Thesis/CNN_Dataset/image'+str(i)+'.png')

def load_positions():
    
    X = []
    y = []

    with open("A0039.txt", 'r') as f:
        for line in f:
            record = line.split(";")
            pieces = [eval(x) for x in record[0:12]]
            piece = [item for sublist in pieces for item in sublist]
            piece = [item for sublist in piece for item in sublist] 
            X.append(piece)
            y.append([float(record[12][:-2])])

    X = np.asarray(X)
    Y = np.asarray(y)

    General_X = []

    for i in X:
        g = i.reshape((12,64))
        tmp = np.zeros(g.shape[1])
        for ind,j in enumerate(g):
            tmp = tmp+j*(ind+1)
        General_X.append(tmp)

    General_X = np.asarray(General_X)

    return General_X

def main():
    positions = load_positions()
    makeImage(positions)

if __name__ == '__main__':
    main()