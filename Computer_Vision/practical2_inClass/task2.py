# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame
# ___________________________________________
# Adam Czajka, Jin Huang, September 2019
# Ryan Karl
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

import cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
import warnings
warnings.filterwarnings("ignore")

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

# Read the image into grayscale
sample = cv2.imread('breakfast2.png')

sample_small = cv2.resize(sample, (640, 480))
cv2.imshow('Grey scale image',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the original image to HSV
# and take H channel for further calculations
sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
sample_h = sample_hsv[:, :, 0]


# Show the H channel of the image
sample_small = cv2.resize(sample_h, (640, 480))
cv2.imshow('H channel of the image',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the original image to grayscale
sample_grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Show the grey scale image
sample_small = cv2.resize(sample_grey, (640, 480))
cv2.imshow('Grey scale image',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after Otsu''s thresholding',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphological operations

# Step 1: Fill the holes.
im_floodfill = binary_image.copy()
h, w = binary_image.shape[ : 2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
binary_image = binary_image | im_floodfill_inv

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after filling',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *** Step 2: It's a good place to apply morphological opening, closing and erosion
# Apply morphological operations
kernel = np.ones((5,5),np.uint8)
binary_image = cv2.erode(binary_image,kernel,iterations = 2)
binary_image = cv2.dilate(binary_image, kernel, iterations = 2)
binary_image = cv2.erode(binary_image,kernel,iterations = 6)


sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after morphological transformation',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find connected pixels and compose them into objects
labels = measure.label(binary_image)
# Calculate features for each object; since we want to differentiate
# between circular and oval shapes, the major and mino_r axes may help; we
# will use also the centroid to annotate the final result
properties = measure.regionprops(labels)

# *** Calculate features for each object:
# - some geometrical feature 1 (dimension 1)
# - some intensity/color-based feature 2 (dimension 2)
features = np.zeros((len(properties), 2))

for i in range(0, len(properties)):
    #Calculate length and width
    features[i, 0] = properties[i].major_axis_length + properties[i].minor_axis_length

    #Calculate hsv intensity at edges
    x, y = properties[i].centroid
    y += + properties[i].major_axis_length * 0.4
    features[i, 1] = sample_h[int(x), int(y)]

# Show our objects in the feature space
plt.plot(features[:, 0],features[:, 1], 'ro')
plt.xlabel('Feature 1: major/minor')
plt.ylabel('Feature 2: mean_intensity')
plt.show()


# *** Choose the thresholds for your features
thrF1 = 120
thrF2 = 40


# *** It's time to classify, count and display the objects
squares = 0
blue_circles = 0
red_circles = 0

fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))


#Apply linear classifier
for i in range(0, len(properties)):
    if (features[i, 0] > thrF1 and features[i, 1] < thrF2):
        squares = squares + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.g', markersize=15)

    if (features[i, 0] < thrF1 and features[i, 1] < thrF2):
        red_circles = red_circles + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.b', markersize=15)

    if (features[i, 0] < thrF1 and features[i, 1] > thrF2):
        blue_circles = blue_circles + 1
        ax.plot(np.round(properties[i].centroid[1]), np.round(properties[i].centroid[0]), '.r', markersize=15)

plt.show()


# That's all! Let's display the result:
print("I found %d squares, %d blue donuts, and %d red donuts." % (squares, blue_circles, red_circles))

