import cv2
import time
import numpy as np
import PoseEstimationModule as pemod

cap = cv2.VideoCapture(0)
detector = pemod.PoseDetector()

pTime = 0
cTime = 0

count_pTime=0
count_cTime=0
counter = 5
is_count_mode = False
is_recorded = False
recLmList = []

x_threshold = 50
y_threshold = 50

def is_record_gesture(lmList):
    x16, y16=lmList[16][1], lmList[16][2]
    x12, y12=lmList[12][1], lmList[12][2]
    # right hand is higher than right sholder
    if y12 - y16 > 0:
        return True
    return False

def is_reset_gesture(lmList):
    x15, y15=lmList[15][1], lmList[15][2]
    x11, y11=lmList[11][1], lmList[11][2]
    # left hand is higher than left sholder
    if y11 - y15 > 0:
        return True
    return False

# countdown until counter will be 0
def rec_count_down():
    global counter, count_pTime, count_cTime
    if is_count_mode == False:
        return -1
    if counter > 0:
        count_cTime = time.time()
        if count_cTime - count_pTime > 1:
            count_pTime = count_cTime
            counter = counter - 1
    return counter

# record one snapshot of lmList
def record_pose(lmList):
    global is_recorded, recLmList
    if is_recorded == False:
        recLmList = lmList
        is_recorded = True

# focus on the landmark 11, 12(left/right sholders), 0(nose)
def judge_poor_posture(lmList):
    global is_recorded, recLmList
    if is_recorded == False:
        return False
    ref_x11, ref_y11 = recLmList[11][1], recLmList[11][2]
    ref_x12, ref_y12 = recLmList[12][1], recLmList[12][2]
    ref_x0, ref_y0 = recLmList[0][1], recLmList[0][2]
    x11, y11 = lmList[11][1], lmList[11][2]
    x12, y12 = lmList[12][1], lmList[12][2]
    x0, y0 = lmList[0][1], lmList[0][2]

    if abs(ref_x11 - x11) > x_threshold or abs(ref_y11 - y11) > y_threshold:
        return True
    elif abs(ref_x12 - x12) > x_threshold or abs(ref_y12 - y12) > y_threshold:
        return True
    elif abs(ref_x0 - x0) > x_threshold or abs(ref_y0 - y0) > y_threshold:
        return True
    return False

while True:
    success, img = cap.read()

    img = detector.find_pose(img)
    h, w, c = img.shape

    lmList = detector.find_position(img)
    if len(lmList) <= 0:
        continue

    if is_reset_gesture(lmList):
        is_count_mode = False
        is_recorded = False
        counter = 5
        cv2.putText(img, "RESET", (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

    if is_record_gesture(lmList):
        is_count_mode = True
        cv2.putText(img, "RECORD MODE", (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

    # params of alpha brend
    alpha = 0.5
    beta = 1 - alpha
    mask = np.zeros(img.shape, np.uint8)

    count = rec_count_down()
    if count > 0:
        img = cv2.addWeighted(img, alpha, mask, beta, 30)
        cv2.putText(img, f'COUNTING: {int(count)}', (int(w/5), int(h/5)), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
    elif count == 0 and is_recorded == False:
        record_pose(lmList)
        img = cv2.addWeighted(img, alpha, mask, beta, 30)
        cv2.putText(img, "POSE Recorded!", (int(w/5), int(h/5) + 10), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    if judge_poor_posture(lmList):
        img = cv2.addWeighted(img, alpha, mask, beta, 128)
        cv2.putText(img, "NEKOZE!!", (int(w/2), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 10)

    # frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
