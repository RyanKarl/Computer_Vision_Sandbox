#Ryan Karl
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
import skimage
import os
import numpy as np
from skimage.color import rgb2gray
from skimage import io
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.filters import threshold_triangle
from skimage.filters import try_all_threshold
import sys

image = io.imread('/Users/ryankarl/Computer_Vision/practical2_atHome/pills.png')
image = rgb2gray(image)
threshold = threshold_otsu(image)
n = 8

image_bin = image > threshold
eroded_image = skimage.morphology.binary_erosion(image_bin)


labels = measure.label(eroded_image, n)
features = measure.regionprops(labels, coordinates='xy')

oval_pills = 0
circle_pills = 0

i = 0
while i < len(features):
    if (features[i].major_axis_length/features[i].minor_axis_length) > 1.5:
        oval_pills += 1
    else:
        circle_pills += 1

    i += 1



print('Number of Oval Pills: ' + str(oval_pills))
print('Number of Circle Pills: ' + str(circle_pills))

