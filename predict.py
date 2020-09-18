import cv2
import numpy as np
from fingertip_finder import Fingertips
from hand_finder import YOLO

fingertips = Fingertips(weights='./weights/Retinanet_Fingertip_Detector.h5')
hand = YOLO(weights='weights/yolo.h5', threshold=0.8)

CAM = cv2.VideoCapture(0)

while CAM.isOpened():
    ret, image = CAM.read()
    # image = cv2.flip(image, 1)
    tl, br = hand.detect(image=image)
    if tl or br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        prob, pos = fingertips.classify(im=cropped_image)
        pos = np.mean(pos, 0)

        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        print(prob)
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        index = 0
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for p in prob:
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12, color=(255, 0, 0), thickness=-2)
            index = index + 2

        # display image
    cv2.imshow('Unified Gesture & Fingertips Detection', image)
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
