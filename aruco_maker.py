import cv2 as cv
from cv2 import aruco
import numpy as np

# Loading the predefined dictionary

Dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# Generating the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.generateImageMarker(Dictionary, 0, 19, markerImage, 1)

# Output the generated image
cv.imwrite("marker0.png", markerImage)




