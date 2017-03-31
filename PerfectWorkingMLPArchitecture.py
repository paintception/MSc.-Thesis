import numpy as np
import keras
import seaborn
import tensorflow as tf 

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
from keras.layers import Dropout
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

print "Reading the data"

X = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/64_125000Positions.npy')
y = np.load('/home/matthia/Desktop/MSc.-Thesis/Datasets/Numpy/64_Representation/64_125000Labels.npy')

print len(X)
print len(y)

dimension_input = X.shape[1]

new_y = np.asarray(y)

encoder = LabelEncoder()
encoder.fit(new_y)
encoded_y = encoder.transform(new_y)
dummy_y = np_utils.to_categorical(encoded_y)		

print "Finished reading the data: ready for the MLP"

def plots(history):
	
	f1 = plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f1.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/ReluAdamAccuracyPlotModel2_'+'.png')
	plt.close()

	f2 = plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Training', 'Validation'], loc='upper left')
	f2.savefig('/home/matthia/Desktop/MSc.-Thesis/Results/Figures/ReluAdamLossPlotModel2_'+'.png')
	plt.close()	

	print "Saving Results"

def baseline_model():

	#sgd = SGD(lr=0.001, decay=1e-15, momentum=0.7, nesterov=False)

	model = Sequential()
	act = keras.layers.advanced_activations.LeakyReLU(alpha=0.01) 
	model.add(Dense(2048, input_dim=dimension_input, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(550, input_dim=dimension_input, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(250, init='normal', activation='relu'))
	model.add(Dropout(0.2))
	#model.add(Dense(250, init='normal', activation='elu'))
	#model.add(Dropout(0.2))
	#model.add(Dense(250, init='normal', activation='elu'))
	#model.add(Dense(250, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='relu'))
	model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

	return model

def main():

	model = baseline_model()
	X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=42)
	tbCallBack = keras.callbacks.TensorBoard(log_dir='/home/matthia/Desktop/logs', histogram_freq=0, write_graph=True, write_images=False)

	history = model.fit(X_train,y_train, nb_epoch=50, batch_size=128, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])
	
	scores = model.evaluate(X,dummy_y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	#model.save('KerasModels/Relu/Model1_'+str(learning_r)+'.h5')
	plots(history)

if __name__ == '__main__':
	main()

