# USAGE
# run classify_dermo_DNN.py --images dermo_dataset/images --masks dermo_dataset/masks

# import the necessary packages

#from __future__ import print_function
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
#from feature.localbinarypatterns import LocalBinaryPatterns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

import numpy as np
import argparse
import glob
import cv2
import csv
from sklearn.datasets import load_files
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# construct a feature vector raw pixel intensities of a color image 3072-D
def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,
	help = "path to the image dataset")
ap.add_argument("-m", "--masks", required = True,
	help = "path to the image masks")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())

#Dictionary to hold key (imagepath) value (category) pair
dict = {} 

str1 = "dermo_dataset/images\\"
str3 =".jpg"

# Read either Caption or Concepts or Both from Training File
with open('ISBI2016_ISIC_Part3_Training_GroundTruth.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	
	for row in reader:
		
		dict [row['imagename']+str3]=row['category']


# grab the image and mask paths
imagePaths = sorted(glob.glob(args["images"] + "/*.jpg"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))

# initialize the list of data and class label targets
data = []
target = []

#print(dict)

# initialize the color histogram image descriptor
desc = RGBHistogram([8, 8, 8])

#26-D LBP Histogram
lbp = LocalBinaryPatterns(24, 8)


# loop over the image and mask paths
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
	
	print (imagePath)  #dermo_dataset/images\ISIC_0011402.jpg
	
	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	#features = image_to_feature_vector(image)
	#data.append(features)
	
	# describe the image
	features = desc.describe(image, mask)
	
	#LBP feature
	lbp_hist = lbp.describe(mask)
	
	
	x = features.tolist()
	y = lbp_hist.tolist()
	x.extend(y)
	combined =np.array(x)    #538-D
			
	#print (combined.shape)
	data.append(combined)    
	
	# update the list of data and targets
	#data.append(features)
	
	cat = imagePath.split("\\")[-1]
	category = dict[cat]
	#print (category)
	
	target.append(category)



# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(target)

data = np.array(data)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

# define the architecture of the network  (81% accuracy obtained!!)
model = Sequential()
model.add(Dense(300, input_dim=538, init="uniform",
	activation="relu"))
model.add(Dense(100, init="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=20,
	verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=20, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

'''
X = np.array(data)

targetNames = np.unique(target)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(target)
encoded_Y = encoder.transform(target)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)

acc = model.score(testRI, testRL)


# define baseline DNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=538, init='normal', activation='relu'))
	model.add(Dense(2, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=20, verbose=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)

estimator.fit(X_train, Y_train)

#estimator.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!


#predictions = estimator.predict(X_test)
#print(predictions)
#print(encoder.inverse_transform(predictions))

# evaluate the classifier
#print(classification_report(Y_test, estimator.predict(X_test),
#	target_names = targetNames))

'''

############# train the classifiers #########################

#-------Random Forest---------------------
'''
model = RandomForestClassifier(n_estimators = 25, random_state = 84)
model.fit(trainData, trainTarget)
'''

#-------Linear SVM---------------------
'''
model = LinearSVC(C=100.0, random_state=42)
#model = svm.SVC(gamma=0.001, C=100)
#model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
model.fit(trainData,trainTarget)
'''

#-------Non Linear SVM---------------------
'''
model = svm.NuSVC(kernel='rbf',nu=0.01)
model.fit(trainData, trainTarget)
'''

'''
#-------K-NN--------------------
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainData,trainTarget)

#image = testData[i]
#prediction = model.predict(image)[0]


acc = model.score(testData, testTarget)
print("[INFO] Classification accuracy: {:.2f}%".format(acc * 100))

# evaluate the classifier
print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))


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