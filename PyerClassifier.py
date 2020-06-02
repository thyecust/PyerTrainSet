# PyerClassfier.py
# by tianh, zeng, lin
# updated on 2020 May 26th

import cv2
# 我把 pimg.py 按照项目风格命名成了 PyerImg.py
import PyerImg as pimg
import numpy as np

def get_circle_imgs(img, k=0.5, enlarge=22, inter=cv2.INTER_CUBIC):
    '''
    输入一张彩色图片, 如
    >>> img = cv2.imread(filename,1)
    >>> count, mats = get_circle_imgs(img)
    返回的格式是 count, mats
    * count: mats 的元素数
    * mats: 一个数组, 每组保存一个图片
    '''
    circles = pimg.get_circles(img)
    circles = np.uint16(np.round(circles))

    count = 0
    mats = []
    for i in range(circles.shape[1]):
        r = k * circles[0][i][2]
        a = [int(circles[0][i][1]-r),int(circles[0][i][1]+r)]
        b = [int(circles[0][i][0]-r),int(circles[0][i][0]+r)]
        mat = img[a[0]:a[1], b[0]:b[1], :]

        fxy = enlarge / r
        mat = cv2.resize(mat, (0,0), \
            fx=fxy, fy=fxy, interpolation=inter)
        count += 1
        mats.append(mat)

    return count, mats

def binarize(img, th=230):
    '''
    输入一副彩色图像, 转换成灰度图像并二值化
    * (gray <= 230) -> 1
    * (gray >  230) -> 0
    '''
    im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, th, 1, cv2.THRESH_BINARY_INV)
    return thresh

def feature(A):  
    midx=int(A.shape[1]/2)+1
    midy=int(A.shape[0]/2)+1
    A1=A[0:midy,0:midx].mean()
    A2=A[midy:A.shape[0],0:midx].mean()
    A3=A[0:midy,midx:A.shape[1]].mean()
    A4=A[midy:A.shape[0],midx:A.shape[1]].mean()
    A5=A[:,0:midx].mean()
    A6=A[:,midx:A.shape[1]].mean()
    A7=A.mean()
    AF=[A1,A2,A3,A4,A5,A6,A7]
    if np.isnan(np.min(AF)):
        return
    else:
        return AF

def incise(im):
    #竖直切割并返回切割的坐标
    a=[]
    b=[]
    if any(im[:,0]==1):
        a.append(0)
    for i in range(im.shape[1]-1):
        if all(im[:,i]==0) and any(im[:,i+1]==1):
            a.append(i+1)
        elif any(im[:,i]==1) and all(im[:,i+1]==0):
            b.append(i+1)
    if any(im[:,im.shape[1]-1]==1):
        b.append(im.shape[1])
    #水平切割并返回分割图片特征
    names=locals()
    afs=[]
    for i in range(len(a)):
        d='null'
        names['na%s' % i]=im[:,range(a[i],b[i])]
        if any(names['na%s' % i][0,:]==1):
            c=0
        elif any(names['na%s' % i][names['na%s' % i].shape[0]-1,:]==1):
            d=names['na%s' % i].shape[0]-1    
        for j in range(names['na%s' % i].shape[0]-1):
            if all(names['na%s' % i][j,:]==0) and any(names['na%s' % i][j+1,:]==1):
                c=j+1
            elif any(names['na%s' % i][j,:]==1) and all(names['na%s' % i][j+1,:]==0):
                d=j+1
        if d!='null': 
            names['na%s' % i]=names['na%s' % i][range(c,d),:]
 
        af = feature(names['na%s' % i])
        if (af):
            afs.append(af)
        #for j in names['na%s' % i]:
         #   print(j)
    return afs

# refactored from feature.py

# 测试

if __name__ == '__main__':
    img = cv2.imread('test.jpg',1)
    total, mats = get_circle_imgs(img)
    #cv2.imshow('',mats[0])
    #cv2.waitKey()
    print(total)
    for i in range(total):
        # cv2.imshow('',mats[i])
        # cv2.waitKey()

        # 测试 binarize()
        b_img = binarize(mats[i])
        # cv2.imshow('',b_img)
        # cv2.waitKey()

        # 测试 feature
        af = incise(b_img)
        print(af)

