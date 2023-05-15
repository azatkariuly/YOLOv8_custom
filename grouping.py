import cv2
import os
import numpy as np

# read in images and store in a list
images = []
for file_name in os.listdir('results'):
    if file_name.endswith('.jpg'):
        image = cv2.imread(os.path.join('results', file_name))
        images.append(image)

# convert images to grayscale
gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

# group images by their average color value
groups = {}
for i, image in enumerate(gray_images):
    avg_color = np.average(image)
    if avg_color not in groups:
        groups[avg_color] = []
    groups[avg_color].append(i)

print(groups)
