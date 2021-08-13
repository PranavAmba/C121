import cv2
import time
import numpy as np

#to save the output in a file

fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

#reading from the webcam

cap=cv2.VideoCapture(0)

#allow the system for 3sec before webcam starts

time.sleep(3)
count=0
background=0

#capture the background in the range of 60

for i in range(60):
    ret,background=cap.read()

background=np.flip(background,axis=1)

while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    count+=1
    img=np.flip(img,axis=1)
    
    #converting the color from bgr to hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #generating mask to detect color
    lower_red=np.array([0,120,50])
    upper_red=np.array([10,255,255])

    mask_1=cv2.inRange(hsv,lower_red,upper_red)

    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])

    mask_2=cv2.inRange(hsv,lower_red,upper_red)

    mask_1=mask_1+mask_2

    #open and dilate the mask image

    mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1=cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    #create an inverted mask to segment out the red color from the frame

    mask_2=cv2.bitwise_not(mask_1)

    #segment the red color part out of the frame using bitwise and the inverted mask
    res1=cv2.bitwise_and(img,img,mask=mask_2)

    #greate image showing static background frame pixels only for the masked region

    res2=cv2.bitwise_and(background,background,mask=mask_1)

    finalOutput=cv2.addWeighted(res1,1,res2,1,0)
    out.write(finalOutput)
    cv2.imshow('magic',finalOutput)
    cv2.waitKey(1)


cap.release()
out.release()
cv2.destroyAllWindows()