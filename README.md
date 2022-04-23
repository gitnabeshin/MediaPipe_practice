# Study of Computer Vision

* https://google.github.io/mediapipe/
* https://www.youtube.com/watch?v=01sAkU_NvOY

## Setup

for iMac

```
% python3 --version
Python 3.9.0
% brew install opencv
% pip3 install opencv-python
% pip3 install mediapipe

% python3 HandTracking.py
```

allow using main camera.

## HandTracking

![IMG_4007](https://user-images.githubusercontent.com/52347942/164245815-a0ff0985-1feb-4a0b-9e42-22ea6d00626b.jpeg)


## Objection

```
% /Applications/Python\ 3.9/Install\ Certificates.command
% python3 -m pip install certifi
% python3 Objectron.py
```

```
objectron = mpObjection.Objectron()
# objectron = mpObjection.Objectron(static_image_mode=True,
#                             max_num_objects=5,
#                             min_detection_confidence=0.5,
#                             model_name='Shoe') 
# model_name {'Shoe', 'Chair', 'Cup', 'Camera'}. Default to Shoe.
```