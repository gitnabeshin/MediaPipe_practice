# 
# https://google.github.io/mediapipe/solutions/face_detection.html
# 
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection
faceDetection = mpFace.FaceDetection()

# cap = cv2.VideoCapture("movie/1.mp4")
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)

            # mpDraw.draw_detection(img, detection)
            bboxRel = detection.location_data.relative_bounding_box
            h ,w, c = img.shape
            bbox = int(bboxRel.xmin * w), int(bboxRel.ymin * h), \
                   int(bboxRel.width * w), int(bboxRel.height * h)
            cv2.rectangle(img, bbox, (255, 255, 0), 2) 
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] -10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,  f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    cv2.waitKey(1)
