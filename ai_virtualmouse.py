import cv2
import numpy as np
import Handtracking as htm
import time
import autopy


wcam , hcam = 640,480
frameR = 100
smoothen = 5
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)
pTime = 0
plocx,plocy = 0,0
clocx,clocy = 0,0
detector = htm.handDetector(maxHands=1)
wscr ,hscr = autopy.screen.size()
print(wscr,hscr)
while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)
        fingers=detector.fingersUp()
        #print(fingers)

        if fingers[1]==1 and fingers[2]==0:
            cv2.rectangle(img,(frameR,frameR),(wcam-frameR,hcam-frameR),(255,0,255),2)
            x3 = np.interp(x1,(frameR,wcam-frameR),(0,wscr))
            y3 = np.interp(y1,(frameR,hcam-frameR),(0,hscr))
            clocx = plocx + (x3-plocx)/smoothen
            clocy = plocy + (y3-plocy)/smoothen

            autopy.mouse.move(wscr-clocx,clocy)

            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocx,plocy = clocx,clocy
        if fingers[1] == 1 and fingers[2] == 1:
            length,img,info= detector.findDistance(8,12,img)
            print(length)
            if length<50:
                cv2.circle(img,(info[4],info[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)

    cv2.imshow('image',img)
    cv2.waitKey(1)
