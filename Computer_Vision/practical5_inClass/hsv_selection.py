# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017

import cv2
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(0)

drawing = False             # True when selecting a Region of Interest
ix, iy = -1,-1              # rectangle starting position
fx, fy = -1,-1              # rectangle final position
still = None

def print_menu():
    print("Options menu:")
    print("esc - Quit program")
    print("s - Snapshot")

def calc_histograms(img):
    color = ['b','g','r']
    colornames = ['Blue channel','Green channel','Red channel']
    plt.figure()
    for i,col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])
    plt.legend(colornames)
    plt.title('RGB Histogram')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    channels = ['Hue','Saturation','Value']
    plt.figure()
    for i,ch in enumerate(channels):
        hist = cv2.calcHist([hsv], [i], None, [256], [0,256])
        plt.plot(hist, color=color[i])
    plt.title('HSV Histogram')
    plt.legend(channels)
    plt.show()

def selectROI(event,x,y,flags,param):
    global ix, iy, fx, fy, drawing, still
    updstill = still.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        # start drawing
        drawing = True
        # save rectangle initial corner
        ix, iy = x, y

    if drawing:
        cv2.rectangle(updstill, (ix, iy),(x,y),(0,255,0),1)

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # save rectangle final corner
        fx, fy = x, y

    cv2.imshow("ROI",updstill)


if __name__ == '__main__':

    print_menu()
    callback = False

    while (True):
        imcrop = None

        retval, img = cam.read()
        res_scale = 0.5             # rescale the input image if it's too large
        img = cv2.resize(img, (0,0), fx=res_scale, fy=res_scale)

        cv2.imshow("Preview", img)

        # handle keyboard commands
        action = cv2.waitKey(1)
        if action == 27:    # escape
            break
        elif action == ord('h'):    # help
            print_menu()
        elif action == ord('s'):    # snapshot
            still = img.copy()
            # r = cv2.selectROI(still)

        # show the snapshot to select the ROI
        if still is not None:
            cv2.imshow("ROI",still)
            if not callback:
                cv2.setMouseCallback("ROI",selectROI)
                callback = True

        # get selected ROI and close the window
        if fx > -1 and fy > -1 and type(still) is not None:
            # selected region of interest
            imcrop = still[iy:fy, ix:fx]
            still = None
            ix, iy, fx, fy = -1, -1, -1, -1
            cv2.destroyWindow("ROI")
            callback = False

        # calculate the histograms for the ROI
        if imcrop is not None:
            cv2.imshow("imcrop",imcrop)
            calc_histograms(imcrop)

    cv2.destroyAllWindows()
