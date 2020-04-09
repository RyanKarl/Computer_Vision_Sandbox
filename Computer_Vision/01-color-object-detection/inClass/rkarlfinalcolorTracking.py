# Ryan Karl
# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017-2019

# Below are your tasks for today. Present your solutions to the instructor / TA in class.
# If you need more time, finish your codes at home and upload them to your SAKAI Dropbox by Wednesday, Sept. 18, 11:59 pm.
#
# Task 1 (2 points):
# - Select one candy that you want to track and set the RGB
#   channels to the selected ranges (using hsvSelection.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space ("ignore" = set the lower bound to 0 and upper bound to 255)? Is so, why?
# - Try to track candies of different colors (blue, yellow, green).
# - What happens when you put two candies of the same color into a video frame?
# - If you have not presented your solution to the instructor in class, upload your code solving this task to your SAKAI Dropbox as colorTracking1.py
#
# Task 2 (1 point):
# - Adapt your code to track multiple candies of *the same* color simultaneously.
# - Upload your solution to your SAKAI Dropbox as colorTracking2.py
#
# Task 3 (2 points):
# - Adapt your code to track multiple candies of *different* colors simultaneously.
# - Upload your solution to your SAKAI Dropbox as colorTracking3.py

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while (True):
    retval, img = cam.read()

    # rescale the input image if it's too large
    res_scale = 0.5
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)



    #######################################################
    # TASK 1:

    # define lower and upper bounds for Blue, Green and Red (NOTE: OpenCV uses BGR instead of RGB)
    # lower = np.array([75, 115, 245])
    # upper = np.array([110, 160, 255])
    # objmask = cv2.inRange(img, lower, upper)

    # Now, comment the RBG-related lines, uncomment the following lines and define lower and upper bounds
    # for Hue, Saturation and Value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bluelower = np.array([105, 0, 0])
    blueupper = np.array([108, 255, 255])
    blueobjmask = cv2.inRange(hsv, bluelower, blueupper)

    yellowlower = np.array([24, 0, 0])
    yellowupper = np.array([26, 255, 255])
    yellowobjmask = cv2.inRange(hsv, yellowlower, yellowupper)

    #lower = np.array([105, 0, 0])
    #upper = np.array([108, 255, 255])
    #objmask = cv2.inRange(hsv, lower, upper)

    objmask = yellowobjmask + blueobjmask

    #######################################################



    # you may use this for debugging
    #cv2.imshow("Binary image", objmask)

    # Resulting binary image may have large number of small objects.
    # We can use morphological operations to remove these unnecessary
    # elements:
    #kernel = np.ones((5,5), np.uint8)
    #objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
    #objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
    #cv2.imshow("Image after morphological operations", objmask)

    # find connected components
    cc = cv2.connectedComponents(objmask)
    ccimg = cc[1].astype(np.uint8)

    # find contours of these objects:

    # use this if you have OpenCV 4.x version:
    contours, hierarchy = cv2.findContours(ccimg,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # use this if you have OpenCV 3.x version:
    # ret, contours, hierarchy = cv2.findContours(ccimg,
    #                                       cv2.RETR_TREE,
    #                                       cv2.CHAIN_APPROX_SIMPLE)

    # You may display the countour points if you want:
    #cv2.drawContours(img, contours, -1, (255,0,0), 3)

    # ignore bounding boxes smaller than "minObjectSize"
    minObjectSize = 20;


    #######################################################
    # TASK 2 tip: think if "if" statement
    # can be replaced by "for" loop
    for val in contours:
    #######################################################

        # use just the first contour to draw a rectangle
        x, y, w, h = cv2.boundingRect(val)
        #######################################################
        # TASK 2 tip: you want to get bounding boxes
        # of ALL contours (not only the first one)
        ######i#################################################


        # do not show very small objects
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(img,            # image
            "Here's my candy!",         # text
            (x, y-10),                  # start position
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.7,                        # size
            (0, 255, 0),                # BGR color
            1,                          # thickness
            cv2.LINE_AA)                # type of line

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action==27:
        break
