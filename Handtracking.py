import cv2
import mediapipe as mp
import time

from pyparsing import results


class HandDetector(object):
    def __init__(self,mode = False,maxHands=2,detectionCon=0.5,trackCon=0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handl_ms in self.results.multi_hand_landmarks:

                self.mpDraw.draw_landmarks(img, handl_ms,self.mpHands.HAND_CONNECTIONS) #draw hand points


        return img
    def findPosition(self,img,handNo = 0,draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]


            for id, lm in enumerate(myHand.landmark): #get the coordinates

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                   cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    ptime = 0
    ctime = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])


        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()