#!/usr/bin/python
#beolvasas
import os
import csv
import numpy as np

y_data=np.load('labels.npy')
EegData=np.load('eegdata.npy')
"""
y_data = [] #desired output

EegData=[] #input data

directory = os.path.join("TrainingsetX") #,"path")
for root,dirs,files in os.walk(directory):
 for file in files:
 	 print("van file")
 	 if file.endswith(".csv"):
 	 	 f=open("TrainingsetX/"+file, 'r')
 
           #  perform calculation
 	 	 print("openinf"+file)
 	 	 if "math" in file:
 	 	 	 y_data=np.append(y_data,1)
 	 	 else:
 	 	 	 y_data=np.append(y_data,0)
 	 	 reader = csv.reader(f)
 	 	 count=0;
 	 	 temp=np.empty((0))
 	 	 for row in reader:
 	 	 	 if count:
 	 	 	 	 temp=np.append(temp,row[2]);
 	 	 	 count=count+1;
 	 	 EegData.append(temp)
 	 	 f.close()

np.save('eegdata.npy',EegData)
np.save('labels.npy',y_data) 
"""
###Training###
import tensorflow as tf
sess = tf.InteractiveSession()
 
MinLength=EegData[0].shape[0]
for Eeg in EegData:
	if MinLength>Eeg.shape[0]:
		MinLength=Eeg.shape[0]
NewEegData=np.zeros((len(EegData),MinLength))
for a in range(len(EegData)):
	NewEegData[a,:]=EegData[a][:MinLength]
EegData=NewEegData

x = tf.placeholder(tf.float32, shape=[len(EegData),EegData[0].shape[0]], name="x_holder")
y = tf.placeholder(tf.int32, shape=y_data.shape[0], name="y_holder")
y_one_hot=tf.one_hot(y,2, name="y_1_hot")
 
W = tf.Variable(tf.zeros([EegData[0].shape[0],2]), name="Weights")
b = tf.Variable(tf.zeros(2), name="bias")
 
NetOut = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=NetOut))
 
optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

init=tf.global_variables_initializer()
NumSteps=100

with tf.Session() as Sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	Sess.run(init)
	Step=1
	while Step<NumSteps:
		_,xe= Sess.run([optimizer,cross_entropy],feed_dict={x: EegData, y: y_data})
		print("Step: "+str(xe))
		Step+=1
	writer.close()



