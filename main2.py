import cv2
import cv2.aruco as aruco

# Initialize video capture from the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def findAruco(img, marker_size=6, total_marker=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_marker}')
    aruco_dict = aruco.getPredefinedDictionary(key)
    aruco_params = aruco.DetectorParameters()
    bbox, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bbox)
    return bbox, ids


while True:
    ret, img = cap.read()
    if not ret:
        break

    # Use the findAruco function to detect and draw ArUco markers
    bbox, ids = findAruco(img)

    # Show the image with detected markers
    cv2.imshow("img", img)

    # Break
