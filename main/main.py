from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot
from numpy.lib.function_base import append
from numpy.lib.npyio import load
from numpy.lib.twodim_base import tri
from pandas.core.indexing import convert_to_index_sliceable
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os

# set folder locations for images
colorFolder = '/Users/jonnylake/Documents/School/418/project/Lego_dataset_1/training/color/'
greyFolder = '/Users/jonnylake/Documents/School/418/project/Lego_dataset_1/training/greyscale/'
testFolder = '/Users/jonnylake/Documents/School/418/project/Lego_dataset_1/testing/'
greyTest = '/Users/jonnylake/Documents/School/418/project/Lego_dataset_1/main/greyscaleTesting'

def load_images_from_folder(folder, greyFolder):
    images = []
    imagesGreyscale = []
    y_array = []

    for filename in os.listdir(folder):
        img1 = image.imread(os.path.join(folder,filename))
        img2 = np.array(Image.open(os.path.join(folder,filename)).convert('L').resize((512,512)))
        gr_im = Image.fromarray(img2).save(os.path.join(greyFolder,filename))

        if "cir" in filename:
            y_array = np.append(y_array,1)
        
        if "rec" in filename:
            y_array = np.append(y_array,2)

        if "squ" in filename:
            y_array = np.append(y_array,3)

        if img2.all() is not None or img1.all() is not None:
            images.append(img1)
            imagesGreyscale.append(img2)

    return imagesGreyscale, y_array

def load_test_images_from_folder(folder, greyFolder):
    images = []
    imagesGreyscale = []

    for filename in os.listdir(folder):
        img1 = image.imread(os.path.join(folder,filename))
        img2 = np.array(Image.open(os.path.join(folder,filename)).convert('L').resize((512,512)))
        gr_im = Image.fromarray(img2).save(os.path.join(greyFolder,filename))

        if img2 is not None:
            images.append(img1)
            imagesGreyscale.append(img2)

    return imagesGreyscale

def evaluate(x_train, y_train, x_test):

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    nsamples, nx, ny = x_train.shape
    nsamples2, nx2, ny2 = x_test.shape
    x_train = x_train.reshape((nsamples,nx*ny))
    x_test = x_test.reshape((nsamples2,nx2*ny2))

    y_train = np.array(y_train)

    classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train).predict(x_test)

    return classifier

# def evaluate(x_train, y_train, x_test):

#     x_train = np.array(x_train)
#     x_test = np.array(x_test)
#     nsamples, nx, ny = x_train.shape
#     nsamples2, nx2, ny2 = x_test.shape
#     x_train = x_train.reshape((nsamples,nx*ny))
#     x_test = x_test.reshape((nsamples2,nx2*ny2))

#     #x_train = x_train.T
#     y_train = np.array(y_train)

#     # train model 1
#     y_train1 = y_train.copy()
#     y_train1[y_train==1]=1
#     y_train1[y_train!=1]=-1
#     y_train1 = y_train1.flatten()

#     log_regress1 = linear_model.RidgeClassifier()
#     log_regress1.fit(x_train,y_train1)
#     y_test1 = (log_regress1.intercept_+np.dot(x_test,log_regress1.coef_.T))/np.linalg.norm(log_regress1.coef_)
#     y_test1 = log_regress1.predict(x_test)

#     # train model 2
#     y_train2 = y_train.copy()
#     y_train2[y_train==2]=1
#     y_train2[y_train!=2]=-1
#     y_train2 = y_train2.flatten()

#     log_regress2 = linear_model.RidgeClassifier()
#     log_regress2.fit(x_train,y_train2)
#     y_test2 = log_regress2.predict(x_test)

#     #train model 3
#     y_train3 = y_train.copy()
#     y_train3[y_train==3]=1
#     y_train3[y_train!=3]=-1
#     y_train3 = y_train3.flatten()

#     log_regress3 = linear_model.RidgeClassifier()
#     log_regress3.fit(x_train,y_train3)
#     y_test3 = (log_regress3.intercept_+np.dot(x_test,log_regress3.coef_.T))/np.linalg.norm(log_regress3.coef_)
#     y_test3 = log_regress3.predict(x_test)

#     return y_train1, y_train2, y_train3, y_test1, y_test2, y_test3

# call label / data manipulation function for circle, rectangle, and squares (training set)
trainingData = load_images_from_folder(colorFolder, greyFolder)
testingData = load_test_images_from_folder(testFolder, greyTest)

x_train = trainingData[0]
y_train = trainingData[1]

x_test = testingData

y_test = evaluate(x_train, y_train, x_test)

print(y_train)
print(y_test)
print("Confusion Matrix:", confusion_matrix(y_train,y_test))
print("Accuracy:", accuracy_score(y_train,y_test)*100, "%")
# print(confusion_matrix(y_test[0], y_test[3]))
# print(confusion_matrix(y_test[1], y_test[4]))
# print(confusion_matrix(y_test[2], y_test[5]))