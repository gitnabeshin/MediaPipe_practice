# 
# https://google.github.io/mediapipe/solutions/pose
# 

import cv2
import mediapipe as mp
import time

class PoseDetector():

    #   def __init__(self,
    #                static_image_mode=False,
    #                model_complexity=1,
    #                smooth_landmarks=True,
    #                enable_segmentation=False,
    #                smooth_segmentation=True,
    #                min_detection_confidence=0.5,
    #                min_tracking_confidence=0.5):
    #     """Initializes a MediaPipe Pose object.
    def __init__( self, 
                  mode=False, 
                  complex=1,
                  smooth=True,
                  en_seg=False,
                  smooth_seg=True,
                  detect_con=0.5,
                  track_con=0.5):
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.en_seg = en_seg
        self.smooth_seg= smooth_seg
        self.detect_con = detect_con
        self.track_con = track_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose( self.mode,
                                 self.complex,
                                 self.smooth,
                                 self.en_seg,
                                 self.smooth_seg,
                                 self.detect_con,
                                 self.track_con)

    def find_pose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)

        #draw land marks
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # for id, lm in enumerate(results.pose_landmarks.landmark):
                #     h, w, c = img.shape
                #     # print(id, lm)
                #     # lm.x =[0.0 - 1.0], lm.y =[0.0 - 1.0]
                #     # map to img area
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                #     # print(id, lm, cx, cy, h, w)
        return img

    def find_position(self, img, draw=True):

        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    # print(id, lm, cx, cy, h, w)

        return lmList

    def find_posture(self, img, draw=True):

        lmList = []
        refList = [0, 11, 12, 23, 24, 16]

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id in refList:
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                    # print(id, lm, cx, cy, h, w)

        return lmList


def main():
    # cap = cv2.VideoCapture("movie/1.mp4")
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = PoseDetector()

    while True:
        success, img = cap.read()

        img = detector.find_pose(img)

        lmList = detector.find_position(img)
        # print(len(lmList))
        if len(lmList) > 0 :
            # Red circle on nose
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 40, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Img", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()