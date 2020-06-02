import cv2
import numpy as np

def get_border(filename, img):
    img2 = img[:,:,2]
    ret,img2th1 = cv2.threshold(img2, 180, 255, cv2.THRESH_TOZERO)
    ret,img2th2 = cv2.threshold(img2th1, 190, 255, cv2.THRESH_TOZERO_INV)
    result = cv2.medianBlur(img2th2, 9)
    whereid = np.where(result > 0)
    whereid = whereid[::-1]
    coords = np.column_stack(whereid)
    (x,y),(w,h),angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle = angle + 90
    vis = img.copy()
    box = cv2.boxPoints(((x,y), (w,h), angle))
    box = np.int0(box)
    cv2.drawContours(vis,[box],0,(0,0,255),2)
    cv2.imwrite(filename, vis)

def get_circles(img):
    '''

    '''
    if(img.shape[0]<1920):
        minRadius=20
    else:
        minRadius=25
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(gimg, 7)
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, \
        1, 10, param1=100, param2=30, minRadius=minRadius, maxRadius=30)
    return circles
