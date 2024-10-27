import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import cv2
import glob
import os
# import skimage as ski
from data_real_image import data_real_image

def flatten(xss):
    return [x for xs in xss for x in xs]


# Fit a KNN pipeline from the master map

def train_ideal_knn(filename):
    xls = pd.read_pickle(filename)
#    print(xls.head())
    y = xls.iloc[:, 0:3]  # column 0 was just numerical indices for some reason
    X = xls.iloc[:, 3:]
    scaler1=StandardScaler()
    Xsc=scaler1.fit_transform(X)
    knn=KNeighborsRegressor(n_neighbors=2, weights='uniform')
    knn.fit(Xsc,y)
    return knn, X.columns


def scaler_from_real_images(realImageDir, columnVals):
    realdata = []
    scaler = StandardScaler()

    images=realImageDir+"*.png"

    for file in glob.glob(images):
        feature_vector, output = data_real_image(file)
        realdata.append(feature_vector)
        print(feature_vector)

    #df=pd.DataFrame(realdata, columns=columnVals)
    #scaledData=scaler.fit(realdata)
    #return scaler

def knn_predict(knn, scaler, feature_vector, columnVals):
    if type(feature_vector)==list:
        fvdf = pd.DataFrame([feature_vector])
        fvdf.columns = columnVals
        fvdf=scaler.transform(fvdf)
        pred=knn.predict(fvdf)
        print(pred)
    else:
        print("Not enough information.")
        pred = False
    return pred

imDir='realImgs'
#print(len(os.listdir(imDir)))
knn, colVals=train_ideal_knn('OctBData.pkl')
scaler_from_real_images(imDir, colVals)