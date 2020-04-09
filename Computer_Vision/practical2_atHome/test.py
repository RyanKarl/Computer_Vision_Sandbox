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

image = io.imread('/Users/ryankarl/Computer_Vision/practical2_atHome/pills.png')
image = rgb2gray(image)
image = skimage.morphology.binary_erosion(image)
thresh = threshold_otsu(image)


# label image regions
label_image = label(thresh)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()



