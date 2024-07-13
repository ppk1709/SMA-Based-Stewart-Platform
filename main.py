import cv2
import cv2.aruco as aruco
import numpy as np

VideoCap=False
cap =  cv2.VideoCapture(2, cv2.CAP_DSHOW)


def findAruco(img,marker_size=6, total_marker=250, draw=True):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{marker_size}X{marker_size}_{total_marker}')
    arucoDict= aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters_create()
    bbox, ids, _=aruco.detectMarkers(gray,arucoDict,parameters= arucoParam)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bbox)
    return bbox, ids




while True:
    _,img= cap.read()
    '''
    else:
        img = cv2.imread('venv/1.png')
    '''
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

    # Create the detector parameters
    parameters = aruco.DetectorParameters_create()


    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers on the image
    image_with_markers = aruco.drawDetectedMarkers(img, corners, ids)

    if cv2.waitKey(1)==113:
        break
    cv2.imshow("img", img)



#parameters = cv2.aruco.DetectorParameters_create()
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)