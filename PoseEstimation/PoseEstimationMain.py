import cv2
import time
import PoseEstimationModule as pemod

cap = cv2.VideoCapture(0)
detector = pemod.PoseDetector()

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    img = detector.find_pose(img)

    lmList = detector.find_position(img)
    # Red circle on nose
    cv2.circle(img, (lmList[0][1], lmList[0][2]), 40, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    cv2.waitKey(1)
