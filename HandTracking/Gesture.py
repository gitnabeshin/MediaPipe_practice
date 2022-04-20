import cv2
import mediapipe as mp
import time
import HandTrackingModule as htmod

pTime = 0
cTime = 0

# for iMac camera
cap = cv2.VideoCapture(0)

detector = htmod.HandDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    # lmList = detector.findPosition(img, 1, False)

    # find first hand 
    lmList = detector.findPosition(img, 0)

    if lmList != 0:
        # print(lmList)
        if lmList is not None and len(lmList) != 0:
            # Check "OK"
            # print(lmList[4], lmList[8])
            x1 ,y1 = lmList[4][1], lmList[4][2]
            x2 ,y2 = lmList[8][1], lmList[8][2]
            if abs(x1 - x2) < 80 and abs(y1 - y2) < 80:
                    h, w, c = img.shape
                    # print(h, w, c)
                    cv2.putText(img, "OK!", (int(w/2), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 30, (0, 0, 255), 20)

    # draw fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, "fps:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
