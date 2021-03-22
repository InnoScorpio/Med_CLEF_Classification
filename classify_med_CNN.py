#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# run classify_med_CNN.py --training MedDB5000/training --testing MedDB5000/testing
# run classify_med_CNN.py --training imageclef2013/training --testing imageclef/testing

import numpy as np
import argparse
import glob
import cv2
from sklearn.datasets import load_files
import os
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.hog import HOG
from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch import dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Input
from keras.utils import np_utils

import matplotlib.pyplot as plt

from keras import backend as K
K.set_image_dim_ordering('th')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True, 
	help="path to the tesitng images")
args = vars(ap.parse_args())

'''
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 3 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons
'''

images = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	#print(imagePath)
	image = cv2.imread(imagePath)
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	imresize = cv2.resize(image, (200, 125))
	images.append(imresize)
	
	
	# extract the label from the image path, then update the
	# label and data lists
	path_list = imagePath.split(os.sep)
	#print path_list[-2]
	labels.append(path_list[-2])
	#labels.append(imagePath.split("\")[-2])
	#labels.append(os.path.split(os.path.dirname(imagePath))[-1]))
	
# grab the unique target names and encode the labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

#print (targetNames)
#print (target)


'''
# loop over the test images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, and describe it
	#print(imagePath)
	image = cv2.imread(imagePath)
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	imresize = cv2.resize(image, (200, 125))
	images.append(imresize)
	
	
	# extract the label from the image path, then update the
	# label and data lists
	path_list = imagePath.split(os.sep)
	#print path_list[-2]
	labels.append(path_list[-2])
	#labels.append(imagePath.split("\")[-2])
	#labels.append(os.path.split(os.path.dirname(imagePath))[-1]))
	
# grab the unique target names and encode the labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
'''


def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = cross_validate(images, target)


# confirm we got our data
print(y_test[0:10])

# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

yy_test = y_test

# one hot encode outputs
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(num_classes)
print("Data normalized and hot encoded.")


def createCNNModel(num_classes):
   
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    epochs = 300  # >>> should be 25+
    lrate = 0.005
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNModel(num_classes)
print("CNN Model created.")

'''
#inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
inp = Input(shape=(125, 200, 3)) 

# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

history = model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

#model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

'''

# fit and run our model
seed = 7
np.random.seed(seed)



history = model.fit(X_train, y_train,                # Train the model using the training set...
          nb_epoch=epochs, batch_size=64,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
         


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

################ evaluate the classifier################################

scores = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
print("done")


# Convert categorial prediction to one single prediction
#print(model.predict(X_test))
proba = model.predict(X_test)
#print(proba.argmax(axis=-1))


print(classification_report(yy_test, proba.argmax(axis=-1),
	target_names = targetNames))


print(metrics.confusion_matrix(yy_test,proba.argmax(axis=-1)))


###Save the model file #################

'''
####Predict accuracy of the test images ##########
# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	imresize = cv2.resize(image, (200, 125))
	
	test_image = np.array(imresize).astype('float32')
	test_image = test_image / 255.0


	prediction = model.predict(test_image)[0]
	print(prediction)
	
	#print(model.score(trainData))
	
	# display the image and the prediction
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
'''

'''
# loop over a sample of the images
for i in np.random.choice(np.arange(0, len(imagePaths)), 3):
	# grab the image and mask paths
	imagePath = imagePaths[i]
	maskPath = maskPaths[i]

	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# describe the image
	features = desc.describe(image, mask)

	#LBP feature
	lbp_hist = lbp.describe(mask)
		
	x = features.tolist()
	y = lbp_hist.tolist()
	x.extend(y)
	combined =np.array(x)
			
	
	# predict the image type
	#dermo = le.inverse_transform(model.predict([features]))[0]
	dermo = le.inverse_transform(model.predict([combined]))[0]
	
	print(imagePath)
	print("I think this dermoscopic image is a {}".format(dermo.upper()))
	cv2.imshow("image", image)
	cv2.waitKey(0)
'''