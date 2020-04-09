# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame
# ___________________________________________
# Adam Czajka, Jin Huang, September 2019

#Ryan Karl
import skimage
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
# print(matplotlib.get_backend())

# Read the image as grayscale
sample = cv2.imread('breakfast1.png', cv2.IMREAD_GRAYSCALE)

sample_small = cv2.resize(sample, (640, 480))
cv2.imshow('Grey scale image',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Binarize the image using Otsu's method
ret1, binary_image = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after Otsu''s thresholding',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *** Here is a good place to apply morphological operations
kernel = np.ones((5,5),np.uint8)
binary_image = cv2.erode(binary_image,kernel,iterations = 2)
binary_image = cv2.dilate(binary_image, kernel, iterations = 4)
binary_image = cv2.erode(binary_image,kernel,iterations = 6)

sample_small = cv2.resize(binary_image, (640, 480))
cv2.imshow('Image after morphological operations',sample_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find connected pixels and groupd them into objects
labels = measure.label(binary_image, 4)

# Calculate features for each object; since we want to differentiate
# between circular and oval shapes, the major and minor axes may help; we
# will use also the centroid to annotate the final result
features = measure.regionprops(labels)
print("I found %d objects in total." % (len(features)))

# In this task it is enough to calculate the ratio
# between tha major and minor axes
his = []
for i in range(0, len(features)):
    if features[i].minor_axis_length > 0:
        his.append(features[i].major_axis_length / features[i].minor_axis_length)

# Now we can look at the histogram to select a global threshold
plt.hist(his)
plt.xlabel("Ratio")
plt.ylabel("Count")
plt.show()

# *** Select a proper threshold
fThr = 1.5



# It's time to classify, count and display the objects
squares = 0
cashews = 0

fig, ax = plt.subplots()
ax.imshow(sample, cmap=plt.cm.gray)

for i in range(0, len(his)):
    if his[i] <= fThr:
        squares = squares + 1
        y, x = features[i].centroid
        ax.plot(x, y, '.g', markersize=10)
    else:
        cashews = cashews + 1
        y, x = features[i].centroid
        ax.plot(x, y, '.b', markersize=10)
plt.show()

# That's all! Let's display the result:
print("I found %d squares, and %d cashew nuts." % (squares, cashews))
