#Create more data
#Structure of the Net

import tensorflow as tf
import numpy as np 
import random
import os
import copy
import time
import seaborn

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

sess = tf.InteractiveSession()

def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1);
    return tf.Variable(initial);

def biasVariable(shape):
    initial = tf.constant(0.1, shape = shape);
    return tf.Variable(initial);
  
nr_epochs = 100000
nr_input = 64;
nr_output = 1;
nr_hidden = 500;
nr_hidden2= 500; 
lr = 0.001;
 
data_input = tf.placeholder(dtype = np.float32, shape = [None, nr_input]);
data_target = tf.placeholder(dtype = np.float32, shape = [None, nr_output]);

W_1 = weightVariable([nr_input, nr_hidden]);
b1  = biasVariable([nr_hidden]);

W_2 = weightVariable([nr_hidden, nr_hidden2]);
b2  = biasVariable([nr_hidden2]);

W_out = weightVariable([nr_hidden2, nr_output]);
b_out = weightVariable([nr_output]);

h1_out = tf.nn.elu(tf.matmul(data_input, W_1) + b1);

h2_out = tf.nn.elu(tf.matmul(h1_out, W_2) + b2);

keep_prob = tf.placeholder(tf.float32)
h2_out = tf.nn.dropout(h2_out, keep_prob)

out = tf.matmul(h1_out, W_out) + b_out;
soft_out=out
out_mse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(out, data_target))))

train = tf.train.GradientDescentOptimizer(lr).minimize(out_mse);
saver = tf.train.Saver();
sess.run(tf.initialize_all_variables());

X = []
y = []

with open("Datasets/Newest.txt", 'r') as f:
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

"""
Complete Representation of the Chess Board
"""

for i in X:
    g = i.reshape((12,64))
    tmp = np.zeros(g.shape[1])
    for ind,j in enumerate(g):
        tmp = tmp+j*(ind+1)
    General_X.append(tmp)

General_X = np.asarray(General_X)

x, X_test2, y, y_test2 = train_test_split(General_X, Y, test_size=0.10, random_state=42)

x_test = X_test2[len(X_test2)/2:]
y_test = y_test2[len(X_test2)/2:]
x_val = X_test2[:len(X_test2)/2]
y_val = y_test2[:len(X_test2)/2]

order = [i for i in xrange(len(x))]

def randomize(inp, out):
    
    n_inp = []
    n_out = []
    
    indxs = [i for i in xrange(len(inp))]
    random.shuffle(indxs)
    
    for ind in indxs:
        n_inp.append(inp[ind])
        n_out.append(out[ind])
    
    return(n_inp, n_out)

errs=[]
Training_Loss = []
Test_Loss = []
Validation_Loss = []

for i in xrange(0, nr_epochs):
   
    train.run(feed_dict = {data_input : x, data_target : y, keep_prob: 0.5});
    
    if i %100 == 0 :
        
        print "Epoch: ", i
        
        _,loss = sess.run([train, out_mse], feed_dict = {data_input : x_val, data_target : y_val, keep_prob: 1})
        
        print 'Loss: ',loss
    
        errs.append(loss)
        
    if i%500==0:
        
        print "testing nn......"
        
        tot=len(y_test)
        errs1=[]
        
        for ind,k in enumerate(y_test):
            ris=soft_out.eval(feed_dict = {data_input : [x_test[ind]], keep_prob: 1})
            ris=ris[0].tolist()
            corr1=k
            er1=abs(ris-corr1)
            errs1.append(er1)
            #print "ris: ", ris
            #print "label: ", k     
    
        print "we got "+str(np.asarray(errs1).mean())+" at epoch "+str(i)

    if i%1000==0 :

        print "full testing nn......"
        
        tot=len(y_test)
        
        corr1=0
        corr2=0
        corr3=0
        
        errs1=[]
        errs2=[]
        errs3=[]

        for ind,k in enumerate(y_test):
            
            ris1=soft_out.eval(feed_dict = {data_input : [x_test[ind]], keep_prob: 1})
            ris2=soft_out.eval(feed_dict = {data_input : [x_val[ind]], keep_prob: 1})
            ris3=soft_out.eval(feed_dict = {data_input : [x[ind]], keep_prob: 1})
            
            corr1=k
            corr2=y_val[ind]
            corr3=y[ind]
            
            er1=abs(ris1-corr1)
            er2=abs(ris2-corr2)
            er3=abs(ris3-corr3)
            
            errs1.append(er1) #Test Set
            errs2.append(er2) #Validation Set
            errs3.append(er3) #Training Set

        Trainingloss = np.asscalar((np.asarray(errs3).mean()))
        Testloss = np.asscalar(np.asarray(errs1).mean())
        Validationloss = np.asscalar(np.asarray(errs2).mean())

        Training_Loss.append(Trainingloss)
        Test_Loss.append(Testloss)
        Validation_Loss.append(Validationloss)

        print(Training_Loss)

        print "we got "+str(np.asarray(errs1).mean())+" at epoch "+str(i)+" on the test set"
        print "we got "+str(np.asarray(errs2).mean())+" at epoch "+str(i)+" on the validation set"
        print "we got "+str(np.asarray(errs3).mean())+" at epoch "+str(i)+" on the data set"

datafile_path = "/home/matthia/Desktop/MSc.-Thesis/Results.txt"
datafile_id = open(datafile_path, 'w+')
data = np.array([Training_Loss, Test_Loss, Validation_Loss])
data = data.T
np.savetxt(datafile_id, data, fmt=['%d','%d','%d'])
datafile_id.close()

plt.plot(Training_Loss, "r", label="Training Loss")
plt.plot(Test_Loss, "b", label="Testing Loss")
plt.plot(Validation_Loss, "g", label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

plt.plot(errs)
plt.title("General Error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()



