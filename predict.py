import cv2
import numpy as np
from fingertip_finder import Fingertips

fingertips = Fingertips(weights='./weights/Retinanet_Fingertip_Detector.h5')

image = cv2.imread('4.jpg')
height, width, _ = (128, 128, 3)

prob, pos = fingertips.classify(im=image)
pos = np.mean(pos, 0)

prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
for i in range(0, len(pos), 2):
    pos[i] = pos[i] * width
    pos[i + 1] = pos[i + 1] * height

index = 0
image = cv2.rectangle(image, (0, 0), (128, 128), (235, 26, 158), 2)
for p in prob:
    if p > 0.5:
        image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=3, color=(255, 0, 0), thickness=-2)
    index = index + 2

cv2.imshow('Unified Gesture & Fingertips Detection', image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
