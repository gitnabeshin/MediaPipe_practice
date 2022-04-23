# 
# https://google.github.io/mediapipe/solutions/face_mesh.html
# 
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpDrawingStyle = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)

# cap = cv2.VideoCapture("movie/1.mp4")
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    if results.multi_face_landmarks:
        for id, face_landmarks in enumerate(results.multi_face_landmarks):
            # print(id, face_landmarks)

            # mpDraw.draw_landmarks(img, face_landmarks)

            mpDraw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=mpDrawingStyle.get_default_face_mesh_tesselation_style())

            # mpDraw.draw_landmarks(
            #     image=img,
            #     landmark_list=face_landmarks,
            #     connections=mpFaceMesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mpDrawingStyle.get_default_face_mesh_contours_style())

            # mpDraw.draw_landmarks(
            #     image=img,
            #     landmark_list=face_landmarks,
            #     connections=mpFaceMesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mpDrawingStyle.get_default_face_mesh_iris_connections_style())

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,  f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Img", img)

    cv2.waitKey(1)
