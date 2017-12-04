#!/usr/bin/python

import os
import csv
import numpy as np

y_data=np.load('labels.npy') #comment for first reading in
EegData=np.load('eegdata.npy')#comment for first reading in
"""
#Uncomment this block for first reading in
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

###Conv initializers###

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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
print(len(EegData))
print((EegData[1].shape[0]))
print( y_data.shape)
EegData=np.asarray(EegData[:])
print(EegData.shape)
 
x = tf.placeholder(tf.float32, shape=[len(EegData),EegData[0].shape[0]], name="x_holder")
y = tf.placeholder(tf.int32, shape=y_data.shape[0], name="y_holder")
y_one_hot=tf.one_hot(y,2, name="y_1_hot")
 
W = tf.Variable(tf.zeros([EegData[0].shape[0],2]), name="Weights")
b = tf.Variable(tf.zeros(2), name="bias")
 
###Convolution functions
x_image = tf.reshape(x, [4,30720,1,1])

W_conv1 = weight_variable([256, 1, 1, 32])
convout=tf.nn.conv2d(x_image, W_conv1,strides=[1,64,1,1], padding='VALID')
b_conv1 = bias_variable([32])
convout=tf.add(convout,b_conv1)
h_conv1 = tf.nn.relu(convout)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 5, 1,1],
                        strides=[1, 2, 1 , 1], padding='VALID')
print h_pool1
W_conv1 = weight_variable([32, 1, 32, 128])
convout=tf.nn.conv2d(h_pool1, W_conv1,strides=[1,8,1,1], padding='VALID')
b_conv1 = bias_variable([128])
convout=tf.add(convout,b_conv1)
h_conv1 = tf.nn.relu(convout)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 1,1],
                        strides=[1, 1, 1 , 1], padding='VALID')
print h_pool1
W_conv1 = weight_variable([7, 1, 128, 256])
convout=tf.nn.conv2d(h_pool1, W_conv1,strides=[1,2,1,1], padding='VALID')
b_conv1 = bias_variable([256])
convout=tf.add(convout,b_conv1)
h_conv1 = tf.nn.relu(convout)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 1,1],
                        strides=[1, 1, 1 , 1], padding='VALID')
print h_pool1

W_fc1 = weight_variable([7 * 256, 2])
b_fc1 = bias_variable([2])

h_pool1_flat = tf.reshape(h_pool1, [-1, 7*256])
h_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1
print h_fc1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=h_fc1))
 
optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

init=tf.global_variables_initializer()
NumSteps=100

with tf.Session() as Sess:
	writer = tf.summary.FileWriter("output_conv", sess.graph)
	Sess.run(init)
	Step=1
	while Step<NumSteps:
		_,xe= Sess.run([optimizer,cross_entropy],feed_dict={x: EegData, y: y_data})
		print("Step: "+str(xe))
		Step+=1
	writer.close()

