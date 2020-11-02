import os
import cv2 as cv
import numpy as np

images = [f for f in os.listdir('Imgs') if f.endswith('.jpg')]

for file in images:
    h, w = cv.imread('Imgs/' + file).shape[:2]
    dummy = np.zeros((h, w), dtype=np.uint8)
    file = file.replace('.jpg', '.png')
    cv.imwrite('GT/' + file, dummy)
