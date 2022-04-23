# 
# https://google.github.io/mediapipe/solutions/objectron
# 
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpObjection = mp.solutions.objectron
objectron = mpObjection.Objectron()
# objectron = mpObjection.Objectron(static_image_mode=True,
#                             max_num_objects=5,
#                             min_detection_confidence=0.5,
#                             model_name='Shoe') 
# modelname {'Shoe', 'Chair', 'Cup', 'Camera'}. Default to Shoe.

# cap = cv2.VideoCapture("movie/1.mp4")
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = objectron.process(imgRGB)

    if results.detected_objects:
        for id, detected_object in enumerate(results.detected_objects):
            # print(id, detected_object)

            mpDraw.draw_landmarks(img, detected_object.landmarks_2d, mpObjection.BOX_CONNECTIONS)
            mpDraw.draw_axis(img, detected_object.rotation, detected_object.translation)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,  f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    cv2.waitKey(1)
