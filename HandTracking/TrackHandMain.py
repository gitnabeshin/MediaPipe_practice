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
    lmList = detector.findPosition(img, 1)
    # if lmList != 0:
    #     print(lmList)
    #     if lmList is not None and len(lmList) != 0:
    #         print(lmList[4])

    # draw fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
