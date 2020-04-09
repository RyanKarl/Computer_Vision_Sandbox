# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017

import cv2

cam = cv2.VideoCapture(0)

print("s - Snapshot")
print("esc - Quit program")

while (True):
    retval, img = cam.read()
    res_scale = 0.5             # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action==27:
        break
    elif action==ord('s'):      # capture
        cap_img = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cap_img,    # image
                    'I can write or draw on an image!!',  # text
                    (10,50),    # start position
                    font,       # font
                    1.0,        # size
                    (0, 255, 0),# BGR color
                    1,          # thickness
                    cv2.LINE_AA )          # type of line
        cv2.line(cap_img,   # image
                 (100,100), # start point
                 (300,300), # end point
                 (0,0,255), # color
                 4)         # thickness

        cv2.imshow("Captured image", cap_img)
