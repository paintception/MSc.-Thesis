import numpy as np
import keras
import seaborn
import pickle

from matplotlib import pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2, activity_l2
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

#opt = SGD(lr=0.01)

#X = []
#y = []

print "Reading the data"

X = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Positions.npy')
y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/Labels.npy')

X = preprocessing.scale(X)
dimension_input = X.shape[1]

new_y = np.asarray(y)

encoder = LabelEncoder()
encoder.fit(new_y)
encoded_y = encoder.transform(new_y)
dummy_y = np_utils.to_categorical(encoded_y)

print "Finished reading the data: ready for the MLP"


#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

def baseline_model():

	model = Sequential()
	model.add(Dense(2048, input_dim=dimension_input, init='normal', activation='relu'))
	model.add(Dense(2048, init='normal', activation='relu'))
	#model.add(Dense(2048, init='normal', activation='tanh'))
	#model.add(Dense(2048, init='normal', activation='relu'))
	#model.add(Dense(2048, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='relu'))
	model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

	return model

def main():

	model = baseline_model()
	X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.10, random_state=42)
	tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/matthia/Desktop/logs', histogram_freq=0, write_graph=True, write_images=False)
	history = model.fit(X_train,y_train, nb_epoch=100000, batch_size=500, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])
	scores = model.evaluate(X,dummy_y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	model.save('/home/matthia/Desktop/MSc.-Thesis/KerasModels/FirstModel.h5')

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	plt.show()
	# "Loss"
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	plt.show()

if __name__ == '__main__':
	main()
