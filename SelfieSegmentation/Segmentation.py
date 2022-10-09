import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

pTime = 0
cTime = 0

# For webcam input:
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
  bg_image = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.2

    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)

    # 1. Gray background ---------------------------------
    # if bg_image is None:
    #   bg_image = np.zeros(image.shape, dtype=np.uint8)
    #   bg_image[:] = BG_COLOR
    # output_image = np.where(condition, image, bg_image)

    # 2. Blur background ---------------------------------
    bg_image = cv2.GaussianBlur(image,(215,215),50)

    # 3. load image --------------------------------------
    # bg_image = cv2.imread('/path/to/image/file')

    output_image = np.where(condition, image, bg_image)

    # Filter -----------------------------
    # output_image = cv2.bilateralFilter(output_image, 5, sigmaColor=50, sigmaSpace=20)
    # output_image = cv2.ximgproc.jointBilateralFilter(image, output_image, 5, sigmaColor=50, sigmaSpace=20)

    # draw fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(output_image, "fps:" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()