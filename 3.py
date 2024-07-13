import cv2 as cv
from cv2 import aruco
import numpy as np
import math
import openpyxl
import matplotlib.pyplot as plt
import time

# Create a new workbook and sheet
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = "Angles Data"
if not sheet['A1'].value:
    sheet.append(["Angle_x", "Angle_y", "Angle_z"])

# Function to plot the plane
def plot_plane(normal, d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    plt.show()

# Function to calculate angles and normal vector relative to the mirror center
def calculate_angle(point1, point2, point3, mirror_center):
    # Translate points relative to the mirror center
    translated_point1 = np.array(point1) - np.array(mirror_center)
    translated_point2 = np.array(point2) - np.array(mirror_center)
    translated_point3 = np.array(point3) - np.array(mirror_center)

    vector1 = translated_point2 - translated_point1
    vector2 = translated_point3 - translated_point1
    normal_vector = np.cross(vector1, vector2)
    d = -np.dot(normal_vector, translated_point1)

    angle_x = 90 - math.degrees(math.acos(normal_vector[0] / np.linalg.norm(normal_vector)))
    angle_y = 90 - math.degrees(math.acos(normal_vector[1] / np.linalg.norm(normal_vector)))
    angle_z = math.degrees(math.acos(normal_vector[2] / np.linalg.norm(normal_vector)))

    return angle_x, angle_y, angle_z, normal_vector, d

# Load calibration data
calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]

# Marker detection settings
MARKER_SIZE = 0.005  # in meters (50 mm)
marker_dict = cv.aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_markers = cv.aruco.DetectorParameters()

# Initialize video capture (use 0 for external camera as per your setup)
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

start_time = time.time()
# Define the mirror center position (10 cm below the camera)
mirror_center = [0, 0, -0.1]  # Assuming the mirror is directly below the camera on the y-axis

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)

    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef)
        # Initialize points for angle calculation
        a1 = a2 = a3 = None

        for i, (ids, corners) in enumerate(zip(marker_IDs, marker_corners)):
            corners = corners.reshape(4, 2).astype(int)
            top_right = corners[0].ravel()
            bottom_right = corners[2].ravel()

            distance = np.linalg.norm(tVec[i][0])
            cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 0.1)
            cv.putText(frame, f"id: {ids[0]} Dist: {round(distance, 2)}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3,
                       (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(frame, f"x:{round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)}", bottom_right,
                       cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv.LINE_AA)

            if ids[0] == 0:
                a1 = [tVec[i][0][0], tVec[i][0][1], tVec[i][0][2]]
                print("Aruco 1 coordinates = ", a1)
            elif ids[0] == 1:
                a2 = [tVec[i][0][0], tVec[i][0][1], tVec[i][0][2]]
                print("Aruco 2 coordinates = ", a2)
            elif ids[0] == 2:
                a3 = [tVec[i][0][0], tVec[i][0][1], tVec[i][0][2]]
                print("Aruco 3 coordinates = ", a3)

        # If all three markers are detected, calculate and log the angles
        if all(a is not None for a in [a1, a2, a3]):
            angle_x, angle_y, angle_z, normal, d = calculate_angle(a1, a2, a3, mirror_center)
            print("Angle with respect to X-axis:", angle_x)
            print("Angle with respect to Y-axis:", angle_y)
            print("Angle with respect to Z-axis:", angle_z)
            sheet.append([angle_x, angle_y, angle_z])
            plot_plane(normal, d)

    else:
        print("Aruco not detected")

    cv.imshow("frame", frame)
    key = cv.waitKey(1)

    if key == ord("q"):
        break

# Save the workbook and release resources
wb.save("angles_data.xlsx")
cap.release()
cv.destroyAllWindows()
