from sklearn import neighbors
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import PyerClassifier as pcla
import joblib
import PyerImg as pimg

def get_point(img):
    #加载识别模型
    clf = joblib.load('PyerPredictModel.pkl')

    #获取点位在图中的坐标
    locationList=[]
    circles = pimg.get_circles(img)
    circles = np.uint16(np.round(circles))
    for i in range(circles.shape[1]):
        flagPosition=[circles[0][i][0],circles[0][i][1]]
        locationList.append(flagPosition)

    #获取点位的数字和点位总数
    classList=[]
    total, mats = pcla.get_circle_imgs(img)
    for i in range(total):
        b_img = pcla.binarize(mats[i])
        af = pcla.incise(b_img)
        aList=af[0]
        classList.append(clf.predict([aList])[0])

    #生成该图点位信息的字典
    result={'length':total,'center':locationList,'class':classList}

    #修正该图点位信息的字典
    recognizeResult=result_correct(result)
    
    return recognizeResult

def result_correct(result):

    total=result['length']
    classList=result['class']

    #点位的数字应小于等于点位总数
    for i in range(total):
        if classList[i] >total:
            classList[i]=total

    #获取应该要有的所有点位数字
    correct=[]
    for i in range(1,total+1):
        correct.append(i)
        
    #获取缺少的点位数字
    for i in classList:
        if correct.count(i)>0:
            correct.remove(i)

    #重复的点位数字改为缺少的点位数字
    for i in range(total):
        for j in range(i+1,total):
            if classList[i]==classList[j]:
                temp=correct[0]
                classList[i]=temp
                correct.remove(temp)

    correctResult={'length':total,'center':result['center'],'class':classList}
    return correctResult

'''
#test
result={'length':3,'center':[0,0,0],'class':[4,4,4}
print(result_correct(result))
'''
img1 = cv2.imread('new.jpg',1)
img2 = cv2.imread('test1.jpg',1)
img3 = cv2.imread('test9.jpg',1)
img4 = cv2.imread('sample.jpg',1)
imgs = [img1,img2,img3,img4]
for img in imgs:
    result=get_point(img)
    print(result)

