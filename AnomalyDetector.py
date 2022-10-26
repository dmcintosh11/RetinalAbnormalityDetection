#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pathlib import Path
from PIL import Image

#%%
cwd = os.getcwd()

dataPath = cwd + '/Data'

trainDataLabels = pd.read_csv(dataPath + '/Training_Set/RFMiD_Training_Labels.csv')

trainDataLabels = trainDataLabels.iloc[:,:2]
trainDataLabels.tail()

#%%

#Adds column to df to have the image file name
trainDataLabels['Image_ID'] = trainDataLabels['ID'].astype(str) + '.png'

#%%
#Iterates through dataframe and splits the normal and abnormal retinal scans into two folders to train autoencoder only on the normal retinas

# iterate over the unique label 
for label in trainDataLabels['Disease_Risk'].unique(): 
    if(label == 0):
        item_name = 'Normal'
    else:
        item_name = 'Abnormal'
    
    # create folder according to the label name 
    item_folder = Path(dataPath + '/Training_Set/' + item_name + 'Training/')
    item_folder.mkdir(parents=True, exist_ok=True)
    
    # iterate over all possible number of unique labels 
    for imageID in trainDataLabels.loc[trainDataLabels['Disease_Risk'] == label]['Image_ID']:
        
        img = Image.open(dataPath + '/Training_Set/Training/' + imageID) # read the image 
        img.save(dataPath + '/Training_Set/' + item_name + 'Training/' + imageID) # and save to target folder 

#%%

#Size of our input images
width = 2144
height = 1424

#Define generators for training, validation and also anomaly data.

batch_size = 64
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    dataPath + '/Training_Set/NormalTraining/',
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='input'
    )

validation_generator = datagen.flow_from_directory(
    dataPath + '/Test_Set/Test/',
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='input'
    )

#anomaly_generator = datagen.flow_from_directory(
#    'cell_images2/parasitized/',
#    target_size=(width, height),
#    batch_size=batch_size,
#    class_mode='input'
#    )

#%%

#Define the autoencoder. 

#Encoder
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(width, height, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

#Decoder
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

#%%

#Fit the model. 
model.fit(
        train_generator,
        steps_per_epoch= 500 // batch_size,
        epochs=1000,
        shuffle = True)