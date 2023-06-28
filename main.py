# -*- coding: utf-8 -*-
"""
@author: Omer
"""

#%% Import

import numpy as np
import tensorflow as tf

#tf.enable_eager_execution ()
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Input, Lambda, Reshape, Add, Multiply, GlobalMaxPooling2D,Subtract
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, Concatenate, concatenate
import matplotlib.pyplot as plt

import pandas as pd
import os as os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from keras.models import Model
from keras import backend 

import cv2

import gc

import visualkeras
import pydot
import graphviz

#%% Section 1: Data Read

#Real Data Below

#images=[]
#for i in range(1,32):
#    data_path = (r'D:\download\val\val_blur_bicubic\X4\000000')
#    data_path2 = data_path + str(f'{i:02}')
#    print(data_path2)
#    for filename in os.listdir(data_path2):
#        image = np.array(cv2.imread(os.path.join(data_path2, filename)))
#        images.append(image)
#images=np.array(images)

#class MyCustomCallback(tf.keras.callbacks.Callback):
#  def on_epoch_end(self, epoch, logs=None):
#    gc.collect()
#
#
##READING x_train
#def read_images(iter):
#    images=[]
#    for i in range(iter,iter+1):
#        data_path = (r"D:\dowload2\train\train_blur_jpeg") + str('\\') + str(f"{i:03}")
#        for filename in os.listdir(data_path):
#            image = np.array(cv2.imread(os.path.join(data_path, filename)))
#            images.append(image)
#    images = np.array(images)
#    
#    x_train = images/255
#    print('x read')
#    del images
#    
#    #READING y_train
#    images=[]
#    for i in range(iter,iter+1):
#        data_path = (r"D:\download\train\train_sharp") + str('\\') + str(f"{i:03}")
#        for filename in os.listdir(data_path):
#            image = np.array(cv2.imread(os.path.join(data_path, filename)))
#            images.append(image)
#    images = np.array(images)
#    
#    y_train = images/255
#    print('y read')
#    
#    x_train = x_train.astype('float32')
#    y_train = y_train.astype('float32')
#    
#    return x_train,y_train

#x_train_test, y_train_test = read_images(0)

#% Params
batch_size = 1
num_epochs = 1

#%
##%% Mnist Data
#
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train/255
#x_test = x_test/255
#
#train_size = 60000
#test_size = 10000
#
#batch_size = 64
#num_epochs = 1

#Add noise
#noise = np.random.normal(0.5,0.05,[train_size,28,28])
#noise2 = np.random.normal(0.5,0.05,[test_size,28,28])
#noise = np.zeros((train_size,28,28))
#noise2 = np.zeros((test_size,28,28))

#y_train = x_train
#y_test = x_test
#
#x_test = np.clip(x_test + noise2, 0, 1)
#x_train = np.clip(x_train + noise, 0, 1)
#
#x_train = x_train.reshape((-1, 28, 28, 1))
#x_test = x_test.reshape((-1, 28, 28, 1)) 
#
#y_train = y_train.reshape((-1, 28, 28, 1))
#y_test = y_test.reshape((-1, 28, 28, 1)) 

#plt.figure()
#plt.imshow(x_train[8,:,:,0])
#plt.figure()
#plt.imshow(y_train[8,:,:,0])
#
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')
#x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')


##%% MNIST DATA BLUR

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

y_train = x_train.copy()
y_test = x_test.copy()

train_size = 60000
test_size = 10000

batch_size = 64
num_epochs = 1

#plt.figure()
#plt.imshow(x_test[5,:,:])

blurred = []
for i in range(train_size):
    x_train[i,:,:] = cv2.blur(x_train[i],(3,3))
    
for i in range(test_size):
    x_test[i,:,:] = cv2.blur(x_test[i],(3,3))
    
x_train = x_train.reshape((-1, 28, 28, 1))

x_test = x_test.reshape((-1, 28, 28, 1)) 

y_train = y_train.reshape((-1, 28, 28, 1))
y_test = y_test.reshape((-1, 28, 28, 1)) 


plt.figure()
plt.imshow(x_test[55,:,:,0])
plt.figure()
plt.imshow(y_test[55,:,:,0])

#%% Section 2: EAM MODULES FOR PAPER KERAS IMPLEMENTATION

def Block1(inputs,bn):     
    x1 = Conv2D(16,(3,3),strides=1,padding="same",dilation_rate=1, activation='relu')(inputs)
    if(bn):
        x1 = BatchNormalization()(x1)
    x1 = Conv2D(16,(3,3),strides=1,padding="same",dilation_rate=2, activation='relu')(x1)
    if(bn):
        x1 = BatchNormalization()(x1)
    
    x2 = Conv2D(16,(3,3),strides=1,padding="same",dilation_rate=3, activation='relu')(inputs)
    if(bn):
        x2 = BatchNormalization()(x2)
    x2 = Conv2D(16,(3,3),strides=1,padding="same",dilation_rate=4, activation='relu')(x2)
    if(bn):
        x2 = BatchNormalization()(x2)

    x = Concatenate(axis=-1)([x1, x2])
    x = Conv2D(16,(3,3),strides=1,padding="same",activation='relu')(x)
    
    x = Add()([x, inputs])
    return x    

def Block2(inputs,bn):     
    x = Conv2D(16,(3,3),strides=1,padding="same",activation='relu')(inputs)
    if(bn):
        x = BatchNormalization()(x)
    x = Conv2D(16,(3,3),strides=1,padding="same")(x)
    x = Add()([x, inputs])
    x = layers.Activation('relu')(x)

    return x    

def Block3(inputs,bn):
    x = Conv2D(16, (3,3),strides=1,padding="same",activation='relu')(inputs)
    if(bn):
        x = BatchNormalization()(x)
    x = Conv2D(16, (3,3),strides=1,padding="same",activation='relu')(x)
    if(bn):
        x = BatchNormalization()(x)
    x = Conv2D(16, (1,1),strides=1,padding="same")(x)
    if(bn):
        x = BatchNormalization()(x)
    x = Add()([x, inputs])
    x = layers.Activation('relu')(x)
    return x

def Block4(inputs,bn):
    x = GlobalAveragePooling2D()(inputs)
#    x = Dense(4,activation='relu')(x)
#    x = Dense(8,activation='sigmoid')(x)
#    x = Multiply()([x, inputs])

    x = tf.expand_dims(x,1)
    x = tf.expand_dims(x,1)         
    x = layers.Conv2D(16, (3,3),padding='same',activation='relu')(x)   #critical choke original = 8
    x = layers.Conv2D(16, (3,3),padding='same',activation='sigmoid')(x)
    x = layers.Multiply()([x, inputs])
        
    return x

    
def EAM(inputs,bn):
    x = Block1(inputs,bn)
    x = Block2(x,bn)
    x = Block3(x,bn)
    x = Block4(x,bn)
    x = Add()([x, inputs])
    return x
    


#%% Section 3: Paper Model Implementation

inputs = keras.Input(shape=(720, 1280, 3))
x0 = layers.Conv2D(filters=32,kernel_size=(2,2), padding="same", strides=(2,2) )(inputs)
x1 = layers.Conv2D(filters=16, kernel_size=(3,3), padding="same")(x0)
x2 = EAM(x1,bn=False)
x3 = EAM(x2,bn=False)
x4 = EAM(x3,bn=False)
x5 = EAM(x4,bn=False)
x6 = Add()([x5, x1])
x6 = layers.Conv2DTranspose(filters=16, strides=(2,2) , kernel_size=(1,1), padding="same")(x6)
x = layers.Conv2D(filters=3, kernel_size=(2,2), padding="same")(x6)

out = Add()([x, inputs])

m1 = keras.Model(inputs, out, name="m1")
m1.summary()
#m1.compile(optimizer='adam', loss='mae', run_eagerly=True, metrics=[tf.keras.metrics.MeanAbsoluteError()])
#
#for iter2 in range(200):
#    iter = iter2 % 40
#    x_train,y_train = read_images(iter)
#    m1.fit(x_train, y_train,validation_split=0.2, epochs=num_epochs, callbacks=[MyCustomCallback()], batch_size=batch_size)
#    del x_train
#    del y_train
#    print(iter2)
#
#x_train,y_train = read_images(50)
#tester = x_train[0]
#tester = tester.reshape(1,720,1280,3)
#preds = m1.predict(tester)

#visualkeras.layered_view(modelorig).show()
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(m1, to_file=dot_img_file, show_shapes=True)

#%% Section 4: DNCNN PAPER IMPLEMENTATION

def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = layers.Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = layers.Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model

DN_model = DnCNN()
DN_model.compile(optimizer='adam', loss='mae')

DN_model.fit(x_train, y_train,validation_split=0.2, epochs=num_epochs, batch_size=batch_size)
DN_model.evaluate(x_test,y_test)


preds2 = DN_model.predict(x_test)

#%% Section 5: UNET PAPER IMPLEMENTATION

def unet(start_neurons):
    input_layer = Input((28, 28, 1))
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    #Middle
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(convm)
    
    #upconv part
    deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding='same', activation='sigmoid')(uconv1)
    model_unet = Model(input_layer, output_layer)
    
    return model_unet

un_model = unet(8)
un_model.compile(optimizer='adam', loss='mae')

#un_model.summary()

#un_model.fit(x_train, y_train,validation_split=0.2, epochs=num_epochs, batch_size=batch_size)
#DN_model.evaluate(x_test,y_test)
#
#preds2 = DN_model.predict(x_test)

#%% Section 6: Plots

#plt.figure()
#plt.title('Input')
#plt.imshow(cv2.cvtColor(tester[0], cv2.COLOR_BGR2RGB))
#plt.savefig("Input_tests2.pdf")
#
#plt.figure()
#plt.title('Preds')
#plt.imshow(cv2.cvtColor(preds[0], cv2.COLOR_BGR2RGB))
#plt.savefig("Pred_tests2.pdf")
#
#plt.figure()
#plt.title('Target')
#plt.imshow(cv2.cvtColor(y_train[0], cv2.COLOR_BGR2RGB))
#plt.savefig("Target_tests2.pdf")


#fig, axs = plt.subplots(3,1)
#plt.suptitle('Images')
#axs[0].set_title('Input')
#axs[0].set(xlabel='x_train')
#axs[0].imshow(cv2.cvtColor(tester[0], cv2.COLOR_BGR2RGB))
#axs[1].set_title('Preds')
#axs[1].set(xlabel='preds')
#axs[1].imshow(cv2.cvtColor(preds[0], cv2.COLOR_BGR2RGB))
#axs[2].set_title('Target')
#axs[2].set(xlabel='y_train')
#axs[2].imshow(cv2.cvtColor(y_train[0], cv2.COLOR_BGR2RGB))
#plt.savefig("Image_tests.pdf")

#%% Section 7: AUTOENCODER TESTS


##%% Autoencoder Example
#input_img = keras.Input(shape=(720, 1280, 3))
#x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
#
#x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
#x = layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
#x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
#decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
##
#autoencoder = keras.Model(input_img, decoded)
##autoencoder.summary()
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=[tf.keras.metrics.MeanAbsoluteError()])
#autoencoder.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
#
#preds21 = autoencoder.predict(x_test)
#
###%% Autoencoder Example 2
#input = keras.Input(shape=(28, 28, 1))
#x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
#x = layers.MaxPooling2D((2, 2), padding="same")(x)
#x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
#x = layers.MaxPooling2D((2, 2), padding="same")(x)
#
## Decoder
#x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
#x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
#x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
#
## Autoencoder
#autoencoder2 = Model(input, x)
#autoencoder2.compile(optimizer="adam", loss="binary_crossentropy",metrics=[tf.keras.metrics.MeanAbsoluteError()])
#autoencoder2.fit(x_train,y_train,validation_data=(x_test,y_test))
#
#preds22 = autoencoder2.predict(x_test)
#
#model = Sequential()
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
# 
#
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
#
#model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
#
#model.fit(x_train, y_train, epochs=10, batch_size=256, shuffle=True, 
#          validation_data=(x_test, y_test))
#
#preds23 = model.predict(x_test)
#
##%% Evaluate
#
#loss1 = m1.evaluate(x_test,y_test)
#loss21 = autoencoder.evaluate(x_test,y_test)
#loss22 = autoencoder2.evaluate(x_test,y_test)
#loss23 = model.evaluate(x_test,y_test)
#
#print('our loss=',loss1, '   their loss21=',loss21,'   their loss22=',loss22, '   their loss23=',loss23)

##%% PSNR
#
#l1 = []
#l2 = []
#l3 = []
#l4 = []
#
#for i in range(test_size):
#    l1.append( tf.image.psnr(preds[i],y_test[i],255).numpy())
#    l2.append( tf.image.psnr(preds21[i],y_test[i],255).numpy())
#    l3.append( tf.image.psnr(preds22[i],y_test[i],255).numpy())
#    l4.append( tf.image.psnr(preds23[i],y_test[i],255).numpy())
#
#psnr1 = np.mean(l1)
#psnr2 = np.mean(l2)
#psnr3 = np.mean(l3)
#psnr4 = np.mean(l4)
#
#print('our loss=',psnr1, '   their loss21=',psnr2,'   their loss22=',psnr3, '   their loss23=',psnr4)

#%% Section 8: More Plots
#
n = 10 
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  # display original + noise 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("others") 
  plt.imshow(tf.squeeze(x_test[i])) 
  bx.get_xaxis().set_visible(False) 
  bx.get_yaxis().set_visible(False) 
  
  # display reconstruction 
  cx = plt.subplot(3, n, i + n + 1) 
  plt.title("our") 
  plt.imshow(tf.squeeze(preds2[i])) 
  cx.get_xaxis().set_visible(False) 
  cx.get_yaxis().set_visible(False) 
  
  # display original 
  ax = plt.subplot(3, n, i + 2*n + 1) 
  plt.title("original") 
  plt.imshow(tf.squeeze(y_test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 
#  
#plt.savefig("Reconstructions3.pdf")


