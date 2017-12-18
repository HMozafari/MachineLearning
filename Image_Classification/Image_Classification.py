
# this code is inspired from: https://www.youtube.com/watch?v=cAICT4Al5Ow
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
# from parser import load_data

# This function breaks down the data folder into "validation" and "train"
def break_down_data_folder_to_train_and_validation(dataFolderPath, trainFolderPath, validFolderPath, trainPercentage):
        # path, dirs, filesInDataFolder = os.walk(dataFolderPath)
        filesInDataFolder = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolderPath):
                filesInDataFolder.extend(filenames)
                break
        numOfFilesInDataFolder = len(filesInDataFolder)
        fileIndex =0
        for i in range(0, numOfFilesInDataFolder):
                if (np.random.random(1) < trainPercentage):
                        os.rename(dataFolderPath +'/' +filesInDataFolder[fileIndex], trainFolderPath +'/' + filesInDataFolder[fileIndex])
                else:
                        os.rename(dataFolderPath + '/' + filesInDataFolder[fileIndex], validFolderPath + '/' + filesInDataFolder[fileIndex])

                fileIndex += 1



# dimensions of our images.
img_width, img_height = 150, 150

datase_path = '/home/shm/MachineLearning/Image_Classification/data'
train_folder_path = '/home/shm/MachineLearning/Image_Classification/data/train'
validation_folder_path = '/home/shm/MachineLearning/Image_Classification/data/validation'



# break_down_data_folder_to_train_and_validation(datase_path, train_data_dir, validation_data_dir, 0.75)

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# Automagically retrieve images and their classes for train and validation sets
# If we put the images in two seperate folders, e.g., data/train/cats and data/train/dogs,
# this flow generator will automatically catagorize them and generate the corresponding batches.
train_generator = datagen.flow_from_directory(
        train_folder_path,
        target_size=(img_width, img_height), #The dimensions to which all images found will be resized.
        batch_size=32,
        class_mode='binary')

print (train_generator)

validation_generator = datagen.flow_from_directory(
        validation_folder_path,
        target_size=(img_width, img_height),
        batch_size=64                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              , # since validation set is smaller we make a bigger batch to have more number of samples per batch
        class_mode='binary')



model = Sequential()
# we make a 3x3 filter that extracts features from the image (convolve on the image)
# in other words, 3x3 which has 32 colors is the input.
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
# reduces the complexity of our NN: it set a window over the feature matrix and
# selcets the max for the features to map them onto a new matrix as Maxpooling
# 1   1   2   4
# 5   6   7   8
# 3   2   1   0  ==> Maxpool(2,2) ==>  6    8
# 1   2   3   4                        3    4

model.add(MaxPooling2D(pool_size=(2, 2)))

############## End of first CNN layer ###############

# repeat the layer
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############## End of second CNN layer ###############

# repeat the layer
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

############## End of third CNN layer ###############

# prevent over-fitting by drop-out
# we flatten out feature map (the layer num. 3)
model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# changes the output of the network the probability since
# sigmoind starts form zero and ends at one.



model.compile(loss='binary_crossentropy', #useful for binary classification
              optimizer='adam', #'rmsprop', # performs gradient descent over batches also we could use adam
              metrics=['accuracy']) # the accuracy not mse

nb_epoch = 30
nb_train_samples = 18000
nb_validation_samples = 6000

(x, y, sample_weight)=model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

model.save_weights('image_classifier_CNN.h5')