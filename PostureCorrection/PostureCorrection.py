import cv2
import time
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

def is_record_posing(lmList):
    x16, y16=lmList[16][1], lmList[16][2]
    x12, y12=lmList[12][1], lmList[12][2]
    # right hand is upper than right sholder
    if y12 - y16 > 0:
        return True
    return False

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

def record_pose(lmList):
    global is_recorded, recLmList
    if is_recorded == False:
        recLmList = lmList
        is_recorded = True

# focus on the landmark 11, 12, 23, 24
def judge_poor_posture(lmList):
    global is_recorded, recLmList
    if is_recorded == False:
        return False
    ref_x11, ref_y11 = recLmList[11][1], recLmList[11][2]
    x11, y11 = lmList[11][1], lmList[11][2]
    ref_x12, ref_y12 = recLmList[12][1], recLmList[12][2]
    x12, y12 = lmList[12][1], lmList[12][2]

    x_threshold = 40
    y_threshold = 40

    if abs(ref_x11 - x11) > x_threshold or abs(ref_y11 - y11) > y_threshold:
        return True
    elif abs(ref_x12 - x12) > x_threshold or abs(ref_y12 - y12) > y_threshold:
        return True
    return False

while True:
    success, img = cap.read()

    img = detector.find_pose(img)

    # lmList = detector.find_posture(img)
    lmList = detector.find_position(img)

    if is_record_posing(lmList):
        is_count_mode = True
        cv2.putText(img, "RECORD MODE", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    count = rec_count_down()
    if count > 0:
        cv2.putText(img, f'COUNTING: {int(count)}', (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    elif count == 0 and is_recorded == False:
        record_pose(lmList)
        cv2.putText(img, "POSE Recorded!", (100, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    if judge_poor_posture(lmList):
        cv2.putText(img, "NEKOZE!!", (200, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 5)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    cv2.waitKey(1)
