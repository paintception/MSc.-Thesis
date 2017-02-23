import numpy
import pandas
import seaborn
import numpy as np
import time
 
from matplotlib import pyplot as plt 
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split
from keras.utils.visualize_util import plot



def AccuracyPlots(history):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def LossPlots(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def run_model():
	# create model
	model = Sequential()
	model.add(Dense(1,input_dim=dimof_input, init='normal', activation='tanh',
		W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(150, init='normal', activation='tanh'))
	#model.add(Dense(150, init='normal', activation='tanh'))
	model.add(Dense(1, init='normal', activation='linear'))
	#A = Adam(lr=0.1, decay=0.1)
	model.compile(loss='mse', optimizer="sgd", metrics=['accuracy'])
	
	return model

def main():
	seed = 7
	numpy.random.seed(seed)

	X = []
	y = []

	with open("Datasets/Newest.txt", 'r') as f:
		for line in f:
			record = line.split(";")
			pieces = [eval(x) for x in record[0:12]]
			piece = [item for sublist in pieces for item in sublist]
			piece = [item for sublist in piece for item in sublist]	
			X.append(piece)
			y.append(float(record[12][:-2]))

	X = np.asarray(X)
	y = np.asarray(y)+1
	print 'converting'
	X=X[:1000]
	y=y[:1000]

	fin=[]
	for i in X:
		g=i.reshape((12,64))
		tmp=np.zeros(g.shape[1])
		for ind,j in enumerate(g):
			tmp=tmp+j*(ind+1)
		fin.append(tmp)
	fin=np.asarray(fin)
	X=fin

	dimof_input = X.shape[1]
	#model = run_model()

	model = Sequential()
	model.add(Dense(1,input_dim=dimof_input, init='normal', activation='tanh'))
	model.add(Dense(2048, init='normal', activation='tanh'))
	model.add(Dense(300, init='normal', activation='tanh'))
	model.add(Dense(150, init='normal', activation='tanh'))
	model.add(Dense(1, init='normal', activation='linear'))
	A = Adam(lr=0.1, decay=0.1)
	model.compile(loss='mse', optimizer=A, metrics=['accuracy'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
	history = model.fit(X_train,y_train, nb_epoch=300, batch_size=5, verbose=1, validation_data=(X_test, y_test))
	scores = model.evaluate(X,y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	AccuracyPlots(history)
	LossPlots(history)

if __name__ == '__main__':
	main()
	#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
