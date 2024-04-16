import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#Sensing env using camera
cap = cv2.VideoCapture(0)

#Finding hand ou of entire frame
detector = HandDetector(maxHands=1)

spacing  =30
imgSize = 400
folderPath = "Data/C"
count = 0


# capturing the hand via camera
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h = hand['bbox']
        
        imgBackground = np.ones((imgSize, imgSize, 3), np.uint8)*255 #<---- Since the image is colored and range of colors is 0-255 i.e. 8 bit numbers
        cropImg = img[y-spacing:y+h+spacing, x-spacing:x+w+spacing]
        
        cropImgShape = cropImg.shape
        
        ratio = h/w
        if ratio>1:
            const = imgSize/h
            wNew = math.ceil(const*w)
            resizeImage = cv2.resize(cropImg, (wNew, imgSize))
            resizeImgShape = resizeImage.shape
            centerGap = math.ceil((imgSize-wNew)/2)
            imgBackground[0:resizeImgShape[0], centerGap:wNew+centerGap] = resizeImage
            
        else:
            const = imgSize/w
            hNew = math.ceil(const*h)
            resizeImage = cv2.resize(cropImg, (imgSize, hNew))
            resizeImgShape = resizeImage.shape
            centerGap = math.ceil((imgSize-hNew)/2)
            imgBackground[centerGap:hNew+centerGap, : ] = resizeImage
            
        cv2.imshow("Image-Background", imgBackground)

        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord(" "):
        count +=1
        cv2.imwrite(f"{folderPath}/image_{time.time()}.jpg", imgBackground)
        print(f"{count} number of image(s) have been stored in the data set")
    

