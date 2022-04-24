import cv2
import mediapipe as mp
import time
import HandTrackingModule as htmod

pTime = 0
cTime = 0

# for iMac camera
cap = cv2.VideoCapture(0)

detector = htmod.HandDetector()

def is_ok(lmList):
    x0 ,y0 = lmList[0][1], lmList[0][2]
    x13 ,y13 = lmList[13][1], lmList[13][2]
    x16 ,y16 = lmList[16][1], lmList[16][2]
    if abs(y0 - y16) < abs(y0 - y13):
        return False
    x17 ,y17 = lmList[17][1], lmList[17][2]
    x20 ,y20 = lmList[20][1], lmList[20][2]
    if abs(y0 - y20) < abs(y0 - y17):
        return False
    x1 ,y1 = lmList[4][1], lmList[4][2]
    x2 ,y2 = lmList[8][1], lmList[8][2]
    if abs(x1 - x2) < 80 and abs(y1 - y2) < 80:
        return True
    return False

def is_v_sign(lmList):
    x4 ,y4 = lmList[4][1], lmList[4][2]
    x16 ,y16 = lmList[16][1], lmList[16][2]
    if abs(x4 - x16) > 30 and abs(y4 - y16) > 30:
        return False
    x20 ,y20 = lmList[20][1], lmList[20][2]
    if abs(x4 - x20) > 40 and abs(y4 - y20) > 40:
        return False
    x0 ,y0 = lmList[0][1], lmList[0][2]
    x5 ,y5 = lmList[5][1], lmList[5][2]
    x8 ,y8 = lmList[8][1], lmList[8][2]
    if abs(y0 - y8) < 1.5 * abs(y0 - y5):
        return False
    x9 ,y9 = lmList[9][1], lmList[9][2]
    x12 ,y12 = lmList[12][1], lmList[12][2]
    if abs(y0 - y12) < 1.5 * abs(y0 - y9):
        return False
    return True


while True:
    success, img = cap.read()

    img = detector.findHands(img)
    # lmList = detector.findPosition(img, 1, False)

    # find first hand 
    lmList = detector.findPosition(img, 0)

    if lmList != 0:
        # print(lmList)
        if lmList is not None and len(lmList) != 0:
            h, w, c = img.shape
            if is_v_sign(lmList):
                cv2.putText(img, "V!", (int(w/2), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 30, (255, 0, 0), 30)
            elif is_ok(lmList):
                cv2.putText(img, "OK!", (int(w/2), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 30, (0, 0, 255), 30)


    # draw fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, "fps:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
