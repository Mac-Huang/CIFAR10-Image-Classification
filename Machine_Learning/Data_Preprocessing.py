import numpy as np
from PIL import Image
import os
import numpy as np
import pandas as pd
from PIL import Image

# Path to the dataset folders
dataset_path = './data/test'
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize list to store the flattened image vectors and labels
data_list = []

# Process each category
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = Image.open(img_path)
        img_array = np.array(img)
        flattened_img_array = img_array.flatten()
        # Append the label and the flattened image array to the list
        data_list.append([category] + flattened_img_array.tolist())

# Convert to DataFrame
# Include the label as the first column
columns = ['label'] + ['pixel'+str(i) for i in range(flattened_img_array.size)]
data_df = pd.DataFrame(data_list, columns=columns)

# Save to CSV file
data_df.to_csv('flattened_images_test.csv', index=False)

# Display the first few rows of the DataFrame
print(data_df.head())