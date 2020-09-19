import cv2
import numpy as np
from fingertip_finder import Fingertips
from hand_finder import YOLO

fingertips = Fingertips(weights='./weights/Retinanet_Fingertip_Detector.h5')
hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
CAM = cv2.VideoCapture(0)

prev_x = 0
prev_y = 0
left = True
movement = []

while CAM.isOpened():
    ret, image = CAM.read()
    if not left:
        image = cv2.flip(image, 1)
    tl, br = hand.detect(image=image)
    if tl or br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        prob, pos = fingertips.classify(im=cropped_image)
        pos = np.mean(pos, 0)

        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        index = 0
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for p in prob:
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])),
                                   radius=12, color=(255, 0, 0), thickness=-2)
            index = index + 2
        if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
            curr_x, curr_y = pos[2], pos[3]
            if prev_x != 0:
                movement.append([[prev_x, prev_y], [curr_x, curr_y]])
            prev_x, prev_y = curr_x, curr_y
            print(movement)
        if np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
            movement = []
    for m in movement:
        (from_x, from_y), (too_x, too_y) = m
        image = cv2.line(image, (from_x, from_y), (too_x, too_y), (0, 255, 0), 2)
    if left:
        image = cv2.flip(image, 1)
    cv2.imshow('Unified Gesture & Fingertips Detection', image)
    k = cv2.waitKey(1)
    if k == ord('s'):
        break
