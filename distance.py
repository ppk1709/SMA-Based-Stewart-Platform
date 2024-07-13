import cv2 as cv
from cv2 import aruco
import numpy as np
import math
import serial
import matplotlib.pyplot as plt
import openpyxl
from mpl_toolkits.mplot3d import Axes3D

wb = openpyxl.Workbook()
# Create a new sheet
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
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    plt.show()


calib_data_path = "calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]



MARKER_SIZE = 0.5  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(1,cv.CAP_DSHOW)

def calculate_angle(point1, point2, point3):
    # Calculate the vectors between the points
    vector1 = [point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]]
    vector2 = [point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]]

    # Calculate the normal vector of the plane
    normal_vector = np.cross(vector1, vector2)
    d = -np.dot(normal_vector, point1)
    # Calculate the angles with respect to the x, y, and z axes
    angle_x = 90- math.degrees(math.acos(normal_vector[0] / math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)))
    angle_y = 90- math.degrees(math.acos(normal_vector[1] / math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)))
    angle_z = math.degrees(math.acos(normal_vector[2] / math.sqrt(normal_vector[0]**2 + normal_vector[1]**2 + normal_vector[2]**2)))
 


    return angle_x, angle_y, angle_z, normal_vector,
while True:
    ret, frame = cap.read()
    a1 = [0,0,0]
    a2 = [0,0,0]
    a3 = [0,0,0]
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Since there was mistake in calculating the distance approach point-outed in the Video Tutorial's comment
            # so I have rectified that mistake, I have test that out it increase the accuracy overall.
            # Calculating the distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            # Draw the pose of the marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)

    if i==0:
        a1 = [tVec[0][0][0], tVec[0][0][1], tVec[0][0][2]]
        print("Aruco 1 coordinates = ", a1)
    elif i==1:
        a1 = [tVec[0][0][0], tVec[0][0][1], tVec[0][0][2]]
        a2 = [tVec[0][0][0], tVec[0][0][1], tVec[0][0][2]]
        print("Aruco 1 coordinates = ", a1)
        print("Aruco 2 coordinates = ", a2)
    elif i==2:
        print("no of aruco",i)
        a1 = [tVec[i-2][0][0], tVec[i-2][0][1], tVec[i-2][0][2]]
        a2 = [tVec[i-1][0][0], tVec[i-1][0][1], tVec[i-1][0][2]]
        a3 = [tVec[i][0][0], tVec[i][0][1], tVec[i][0][2]]
        print("Aruco 1 coordinates = ", a1)
        print("Aruco 2 coordinates = ", a2)
        print("Aruco 3 coordinates = ", a3)
        angle_x, angle_y, angle_z, normal, d = calculate_angle(a1, a2, a3)

        print("Angle with respect to X-axis:", angle_x)
        print("Angle with respect to Y-axis:", angle_y)
        print("Angle with respect to Z-axis:", angle_z)
        sheet.append([angle_x, angle_y, angle_z])

        # Plot the plane
        #plot_plane(normal, d)
        # Show the plot





    else:
        print("aruco not detected")


    key = cv.waitKey(1)
    if key == ord("q"):
        break

wb.save("new.xlsx")
cap.release()
cv.destroyAllWindows()