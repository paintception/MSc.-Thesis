import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("Datasets/Data.csv")
dataset = dataframe.values

for y in range(len(dataset)):
	for x in range(len(dataset[y])):
		v = dataset[y][x]
		if isinstance(v, basestring):
			dataset[y][x] = int(v, 2)

# split into input (X) and output (Y) variables
X = dataset[:,0:12]
y = dataset[:,13]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

def run_model():
	# create model
	model = Sequential()
	model.add(Dense(768, input_dim=12, init='normal', activation='relu'))
	model.add(Dense(2048, init='normal', activation='relu'))
	model.add(Dense(2048, init='normal', activation='relu'))
	model.add(Dense(2048, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='linear'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model

def main():
	model = run_model()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	model.fit(X_train,y_train, nb_epoch=10000, batch_size=5, verbose=1, validation_data=(X_test, y_test))
	scores = model.evaluate(X,Y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__ == '__main__':
	main()
#estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=1000, batch_size=5, verbose=0)
#kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
