import numpy as np

# Load the NPZ file
data = np.load('your_file.npz')

# List all arrays in the NPZ file
print(data.files)

# Access an array
array_1 = data['array_name_1']
array_2 = data['array_name_2']

# Print the arrays to verify
print(array_1)
print(array_2)
