import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Function to calculate the equation of a plane given three points
def plane_equation(point1, point2, point3):
    vector1 = point2 - point1
    vector2 = point3 - point1
    normal = np.cross(vector1, vector2)
    d = -np.dot(normal, point1)
    return normal, d


# Function to plot the plane
def plot_plane(normal, d):
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5)


# Get input coordinates from the user
x_coords = []
y_coords = []
z_coords = []

for i in range(3):
    x = float(input(f"Enter x-coordinate for point {i + 1}: "))
    y = float(input(f"Enter y-coordinate for point {i + 1}: "))
    z = float(input(f"Enter z-coordinate for point {i + 1}: "))

    x_coords.append(x)
    y_coords.append(y)
    z_coords.append(z)

# Convert coordinates to NumPy arrays
point1 = np.array([x_coords[0], y_coords[0], z_coords[0]])
point2 = np.array([x_coords[1], y_coords[1], z_coords[1]])
point3 = np.array([x_coords[2], y_coords[2], z_coords[2]])

# Calculate the plane equation
normal, d = plane_equation(point1, point2, point3)
print("Equation of the plane:")
print(f"{normal[0]}*x + {normal[1]}*y + {normal[2]}*z + {d} = 0")

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot the points
ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o')

# Plot the plane
plot_plane(normal, d)

# Show the plot
plt.show()
