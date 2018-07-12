#!/usr/bin/python

import numpy as np
'''
x_data=np.load('../adatfeldolg/eeg2.npy')
y_data=np.load('../adatfeldolg/imgresult.npy')[:,1]

max=len(x_data[0])
for i in range(1,len(x_data)):
	if len(x_data[i]) > max:
		max = len(x_data[i])

for i in range(0,len(x_data)):
	x_data[i]=np.pad(x_data[i],((0,max-len(x_data[i]))),mode='constant', constant_values=0)
temp=x_data[0]
for i in range(1,len(x_data)):
	temp = np.row_stack((temp,x_data[i]))
x_data=temp
x_data=np.array(x_data).astype('float32')
#x_data=np.asarray(x_data)
x_data=x_data.reshape((len(x_data),max,1))
print x_data
print type(x_data)
print max
print x_data.shape


#print str(len(x_data))+' '+str(len(y_data))
#print y_data

print 'dimenzio:' + str(x_data.ndim)

#x_data=x_data.reshape(x_data.shape[0],1,)

#np.savetxt("eeg_data.csv", data, delimiter=",")
#np.savetxt("y_data.csv", y_data, delimiter=",")

np.save('eegdata',x_data)
'''
x_data=np.load('eegdata.npy')
y_data=np.load('imgresult.npy')[:,1]

print x_data

from keras.models import Sequential
from keras import layers
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(y_data, num_classes=3)

print categorical_labels

'''
from keras import backend as K
print(K.image_data_format()) # print current format
K.set_image_data_format('channels_last') # set format
'''

# gradient descent stochastic?
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


model=Sequential() #egymas utan jovo layerek
model.add(layers.Conv1D(32,9,activation='relu',input_shape=(len(x_data[0]),1)))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(64,9,activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32,5,activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(3,activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
plot_model(model,show_shapes='True', to_file='model.png')

history=model.fit(x_data,categorical_labels,epochs=10,batch_size=10,validation_split=0.2)

#rmsprop grad descent fajta

import matplotlib.pyplot as plt

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf

acc=history.history['acc']
val_acc=history.history['val_acc']

plt.plot(epochs,acc,'bo',label='Training loss')
plt.plot(epochs,val_acc,'b',label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
