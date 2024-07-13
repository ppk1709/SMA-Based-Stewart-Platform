import numpy as np

# Load the .npz file
data = np.load('calib_data/MultiMatrix.npz')

# Access the contents
for key in data.keys():
    print(f"Array name: {key}")
    print(f"Array data: {data[key]}")
