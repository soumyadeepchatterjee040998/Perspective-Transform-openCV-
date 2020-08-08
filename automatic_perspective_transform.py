import cv2
import numpy as np
import time

path = 'photo.jpg'
cap = cv2.VideoCapture(0)
time.sleep(3)
videocap = True

scale = 2
w=210*scale
h=297*scale

def reorder(pnts):
    newpnts = np.zeros_like(pnts, dtype = "float32")
    add = pnts.sum(1)
    newpnts[0] = pnts[np.argmin(add)]
    newpnts[3] = pnts[np.argmax(add)]
    diff = np.diff(pnts,axis=1)
    newpnts[1]=pnts[np.argmin(diff)]
    newpnts[2] = pnts[np.argmax(diff)]
    return newpnts


def perimg(img,pnts,w,h):
    pad=20
    pnts = np.reshape(pnts,(4,2))
    pnts = reorder(pnts)
    p1 = np.float32(pnts)
    p2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(p1,p2)
    img = cv2.warpPerspective(img,matrix,(w,h))
    img = img[pad:img.shape[0]-pad,pad:img.shape[1]-pad]
    return img


def getcontour(img,treshold=[100,100],draw_cont=False,draw_canny=False,minarea=1000,filter_=4):
    img_ = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),1)
    img = cv2.Canny(img,treshold[0],treshold[1])
    if draw_canny==True:
        cv2.imshow('canny',img)
        cv2.waitKey(1)
    kernal = np.ones((5,5))
    img = cv2.dilate(img,kernal,iterations=3)
    img = cv2.erode(img,kernal,iterations=1)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= minarea:
            peri = cv2.arcLength(contour,closed=True)
            dp = cv2.approxPolyDP(contour,0.02*peri,True)
            bbox = cv2.boundingRect(dp)
            if len(dp)==filter_:
                final.append([dp,area,contour,bbox])     
    final = sorted(final,key=lambda x:x[1],reverse=True)
    if draw_cont == True:
        for cont in final:
            cv2.drawContours(img_,cont[2],-1,(0,0,255),3)     
    return final

while(True):
    if videocap:
        flag,img = cap.read()
    else:
        img = cv2.imread(path)
    final= getcontour(img,minarea=3000,draw_cont=False,draw_canny=False,filter_=4)
    if len(final)!=0:
        pnts = final[0][0]
        img = perimg(img,pnts,w,h)
        
    cv2.imshow('output',img)
    k = cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()