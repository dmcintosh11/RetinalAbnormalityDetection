#%%
#The only model that I could somewhat get to work was mod2 in this file since my laptop was not nearly powerful enough to handle the proper architecture of the model

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
#import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#%%
cwd = os.getcwd()

dataPath = cwd + '/Data/'

width = 2144
height = 1424


train = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_generator = train.flow_from_directory(
    dataPath + 'Training_Set/Images',
    target_size=(width, height),
    batch_size=16,
    class_mode='binary',
    subset = 'training'
    )

validation_generator = train.flow_from_directory(
    dataPath + 'Training_Set/Images',
    target_size=(width, height),
    batch_size=16,
    class_mode='binary',
    subset = 'validation'
    )

#%%


validation_generator.class_indices
train_generator.class_indices

#%%

model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(width,height,3)))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
model.add(keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Abormal or 1 for Normal 
#Here we use sigmoid activation function which makes our model output lie between 0 and 1
model.add(keras.layers.Dense(1,activation='sigmoid'))

#%%
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#%%
mod2 = keras.Sequential()

# Convolutional layer and maxpool layer 1
mod2.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(width,height,3)))
mod2.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
mod2.add(keras.layers.Flatten())

mod2.add(keras.layers.Dense(1,activation='sigmoid'))

mod2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'AUC'])

#%%

mod3 = keras.Sequential()

mod3.add(keras.layers.Dense(32, activation="relu"))
mod3.add(keras.layers.Dense(16, activation="relu"))

mod3.add(keras.layers.Dense(1,activation='sigmoid'))

mod3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'AUC'])


#%%
mod4 = keras.Sequential()

# Convolutional layer and maxpool layer 1
mod4.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(width,height,3)))
mod4.add(keras.layers.MaxPool2D(2,2))

# This layer flattens the resulting image array to 1D array
mod4.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and Rectified Linear Unit activation function 
mod4.add(keras.layers.Dense(512,activation='relu'))

# Output layer with single neuron which gives 0 for Cat or 1 for Dog 
#Here we use sigmoid activation function which makes our model output to lie between 0 and 1
mod4.add(keras.layers.Dense(1,activation='sigmoid'))

mod4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', 'AUC'])


#%%


#steps_per_epoch = train_imagesize/batch_size

mod2.fit(train_generator,
         steps_per_epoch = 96,
         epochs = 3,
         validation_data = validation_generator
         )

#%%
def predictImage(filename):
    img1 = image.load_img(filename,target_size=(width,height))
    
    plt.imshow(img1)
 
    Y = image.img_to_array(img1)
    
    X = np.expand_dims(Y,axis=0)
    val = mod2.predict(img1)
    print(val)
    if val == 1:
        
        plt.xlabel("Normal",fontsize=30)
        
    
    elif val == 0:
        
        plt.xlabel("Abnormal",fontsize=30)

#%%
predictImage(dataPath + 'Training_Set/Images/Normal/274.png')

#%%
trainpred = mod2.predict(train_generator)

#%%
for i in trainpred:
    print(i)

#%%
print(sum(np.around(trainpred,0)))

#%%
batch = next(train_generator)

#%%
plt.imshow(batch[0][0])

#%%
item = batch[0][0]
#plt.imshow(item)
val = mod2.predict(item)
plt.imshow(item)
print(val)
if val == 1:
        
    plt.xlabel("Normal",fontsize=30)
        
    
elif val == 0:
        
    plt.xlabel("Abnormal",fontsize=30)
#%%
!mkdir -p saved_model
mod2.save('saved_model/my_model')
