# USAGE
# run recognize_med.py --training imageclef2013/training --testing imageclef/testing
# run recognize_med.py --training MedDB5000/training --testing MedDB5000/testing

# import the necessary packages
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

#from imutils import paths
import paths
import argparse
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True, 
	help="path to the tesitng images")
args = vars(ap.parse_args())


##################Visual Feature Extarction #####################################
# initialize the HOG descriptor  1800-D
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)

# 512-D Histogram
rgb = RGBHistogram([8, 8, 8])

#26-D Histogram
lbp = LocalBinaryPatterns(24, 8)

# initialize the color descriptor (5-region)
cd = ColorDescriptor((8, 12, 3))

data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	#print(imagePath)
	image = cv2.imread(imagePath)
	
	# Resize image to 100 x 100 
	image = cv2.resize(image, (100, 100)) 
		
	# RGB Histogram
	rgb_hist = rgb.describe(image)
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#LBP feature
	lbp_hist = lbp.describe(gray)
		
	gray = dataset.deskew(gray, 100)
	gray = dataset.center_extent(gray, (100, 100))
		
	#HOG feature	
	hog_hist = hog.describe(gray)
	
	# describe the color of 5 regions in image
	#features = cd.describe(image)
	
	x = rgb_hist.tolist()
	y = lbp_hist.tolist()
	z = hog_hist.tolist()
	x.extend(y)
	x.extend(z)
	
	combined =np.array(x)
		
	#print (combined.shape)
	data.append(combined)

	
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
labels = le.fit_transform(labels)


########## construct the training and testing splits ########################

(trainData, testData, trainTarget, testTarget) = train_test_split(data, labels,
	test_size = 0.3, random_state = 42)


#################Classification##########################################

#Random Forest classifier
#model = RandomForestClassifier(n_estimators = 25, random_state = 84)
#model.fit(trainData, trainTarget)


# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(trainData, trainTarget)

'''
#SVM in OpenCV 3.1.0 for Python
SVM = cv2.ml.SVM_create()
SVM.setKernel(cv2.ml.SVM_LINEAR)
SVM.setP(0.2)
SVM.setType(cv2.ml.SVM_EPS_SVR)
SVM.setC(1.0)

#training
SVM.train_auto(trainData, cv2.ml.ROW_SAMPLE, trainTarget)
#predict
#output = SVM.predict(samples)[1].ravel()
'''

##Another Way
'''
trainingDataMat = np.array(trainData, np.float32)
labelsMat = np.array(trainTarget, np.int32)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setDegree(0.0)
# svm.setGamma(0.0)
# svm.setCoef0(0.0)
# svm.setC(0)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labelsMat)

sample_data = np.array(testData, np.float32)
response = svm.predict(sample_data)
#response = np.array(response, np.int32)
print response
'''

#The following line is modified for OpenCV 3.0
###KNN
'''
knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

print ("results: ", results,"\n")
print ("neighbours: ", neighbours,"\n")
print ("distances: ", dist)

#knn = cv2.KNearest()
#knn.train(trainData, trainTarget)
#ret, result, neighbours, dist = knn.find_nearest(testData, k=5)

#correct = np.count_nonzero(result == labels)
#accuracy =  correct*100.0/10000
#print accuracy
#print("The number of correct (KNN) is a {}".format(correct.upper()))
'''

###Bayes Classifier
#clf = MultinomialNB().fit(tf_idf_matrix,data_set3)
#model = MultinomialNB().fit(trainData, trainTarget)

'''
###ANN
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

ann.train(np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], dtype=np.float32),
  cv2.ml.ROW_SAMPLE,
  np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32))

print ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32))
'''

################ evaluate the classifier################################
acc = model.score(testData, testTarget)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))


'''
# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	hist = hist.reshape(1,-1)
	prediction = model.predict(hist)[0]
	print(prediction)
	
	#print(model.score(trainData))
	
	# display the image and the prediction
	cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
'''