import numpy as np
import cv2

cv2.namedWindow("Image")
(w,h) = (250,350)

img = cv2.imread('image1.jpg')

img_resize = cv2.resize(img,(w,h))


copy = img_resize.copy()
ref_ptr = []
count = 0

def fun(event,x,y,flag,param):
    global ref_ptr,count
    if event == cv2.EVENT_LBUTTONDOWN and count<=4:
        ref_ptr.append((x,y))
        cv2.circle(copy,(x,y),5,(0,255,0),cv2.FILLED)
        count += 1
        
cv2.setMouseCallback("Image",fun)
while True:
    cv2.imshow("Image",copy)
    key = cv2.waitKey(1)
    if key==ord('r'):
        ref_ptr = []
        count=0
        copy = img_resize.copy()
    elif key==ord('c'):
        break
        
print(ref_ptr)

ptr1 = np.float32(ref_ptr)
ptr2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

op = cv2.getPerspectiveTransform(ptr1,ptr2)
img_per = cv2.warpPerspective(img_resize,op,(w,h))

kernal =  np.float32([[-1,-1,-1],
                    [-1,9,-1],
                    [-1,-1,-1]])
img_filter = cv2.filter2D(img_per,-1,kernal)
img_gray = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY )
img_equ = cv2.equalizeHist(img_gray)
img_gauss = cv2.GaussianBlur(img_gray,(3,3),0)

cv2.imshow("Image1",img_resize)

array = cv2.hconcat([img_per,img_filter])
cv2.imshow("Image2",array)


array = cv2.hconcat([img_gray,img_equ,img_gauss])
cv2.imshow("Image3",array)
cv2.waitKey(0)

cv2.destroyAllWindows()