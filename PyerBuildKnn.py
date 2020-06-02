from sklearn import neighbors
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PyerClassifier as pcla
import joblib

flagName=[]
flagFeature=[[],[],[],[],[],[],[]]
flagType=[]
for i in range(1,5):
    for j in range(0,7):
        flagType.append(i)
        flagname=str(i)+"-"+str(j)
        filestr=str(i)+"/"+flagname
        flagName.append(flagname)
        filename="./"+filestr+".jpg"
        img = cv2.imread(filename,1)
        b_img = pcla.binarize(img)
        af = pcla.incise(b_img)
        for k in range(0,7):
            flagFeature[k].append(af[0][k])
#print(flagFeature)
#print(flagName)
#print(flagType)

data=pd.DataFrame({'name':flagName,
                   'feature1':flagFeature[0],
                   'feature2':flagFeature[1],
                   'feature3':flagFeature[2],
                   'feature4':flagFeature[3],
                   'feature5':flagFeature[4],
                   'feature6':flagFeature[5],
                   'feature7':flagFeature[6],
                   'type':flagType})

knn=neighbors.KNeighborsClassifier()

knn.fit(data[['feature1','feature2','feature3','feature4','feature5','feature6','feature7']],data['type'])
joblib.dump(knn, 'PyerPredictModel.pkl')
