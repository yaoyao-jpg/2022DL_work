import time

import cv2
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

print('Starting to get data...')
a = time.time()

data = pd.read_csv('../objects.csv')


def path2pic(path) -> np.ndarray:
    img = cv2.imread(f'../data/{path}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.swapaxes(img, 0, 2)
    if np.shape(img) != (3, 64, 64):
        print(type(img))
        print(np.shape(img))
        print(img)
        print('Error!')
        exit(0)
    return img / 256


data['Class'] = pd.factorize(data.Class)[0]

X = data['objID'].apply(path2pic).values
Y = data.Class.values

X_ = np.zeros((36888, 3, 64, 64))

for index, i in enumerate(X):
    X_[index] = i

X = X_

print('------------------------------------------')
print('所用时间：' + str(time.time() - a) + 's')
print('------------------------------------------')


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, train_size = 0.85)

def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))  # 一维化
    mean_image = np.mean(x_train, axis=0)  # 求每一列均值。即求所有图片每一个像素上的平均值
    return mean_image

def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(np.float64)
    x_test -= mean_image  # Subtract the mean from the graph, and you get zero mean graph
    return x_test

mean_image = getXmean(xtrain)
xtrain = centralized(xtrain, mean_image)

mean_image = getXmean(xtest)
xtest = centralized(xtest, mean_image)

model = SVC()
x=time.time()
model.fit(xtrain,ytrain)
print(time.time()-x)
ypred= model.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ypred, ytest))
#Liner_SVM:0.957
#SVM:0.977
#KNN:0.965



