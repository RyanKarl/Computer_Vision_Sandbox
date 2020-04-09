# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ______________________________________________________________________
# Jin Huang, Aakash Bansal, Adam Czajka, September 2018 -  November 2019


########################################
# Import libraries
########################################
import cv2
import numpy as np
from sys import platform as sys_pf
import warnings
warnings.filterwarnings("ignore")
from numpy.linalg import multi_dot
from numpy.linalg import lstsq

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
plt.plot()

#######################################
# Define parameters
#######################################
# Set the value range for the HSV
h_lower, h_upper = 0, 27
s_lower, s_upper = 130, 220
v_lower, v_upper = 130, 255

# Setup for displaying results
res_scale = 0.5
minObjectSize = 10
frame_step = 1

# Kalman filter parameters
# State, in this case rigged as [x y dx dy]
kal_x = np.matrix([0, 0, 0, 0])
# Filter confidence (state covariance)
kal_P = np.identity(4)
kal_C = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

# TRANSITION MATRIX
dt = 1
kal_A = np.matrix([[1, 0, dt, 0],
                    [0, 1,  0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# NOISE
kal_R = 0.1 * np.identity(2) # measurement noise
kal_Q = 0.1 * np.identity(4) # process noise


def kalman_predict(A, X):
    return np.dot(A, X.getH())

def kalman_update(kal_P_init, predicted, measured):
    Pk = multi_dot([kal_A, kal_P_init, kal_A.getH()]) + kal_Q

    K_left_matrix = np.dot(Pk, kal_C.getH())
    K_right_matrix = multi_dot([kal_C, Pk, kal_C.getH()]) + kal_R
    K = lstsq(K_left_matrix.T, K_right_matrix.T)[0]
    kal_x = (np.matrix(predicted) +
             np.dot(K, (np.matrix(measured) -
                        np.dot(kal_C, np.matrix(predicted))))).getH()
    kal_P = np.dot((np.identity(4) - np.dot(K, kal_C)), Pk)

    return Pk, K, kal_x, kal_P


if __name__ == '__main__':
    # Use live stream video
    cam = cv2.VideoCapture(0)
    print("Start tracking...")


    while (True):
        if frame_step != 1:
            for i in range(frame_step):
                ret, frame = cam.read()
        else:
            ret, frame = cam.read()

            # Resize the frames here to make the process faster
            frame_current = cv2.resize(frame, (0, 0),
                                       fx = res_scale,
                                       fy = res_scale)

            # Use the tracking method from practical 1
            hsv = cv2.cvtColor(frame_current, cv2.COLOR_BGR2HSV)
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, s_upper, v_upper])
            objmask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((5, 5), np.uint8)
            objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
            objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)

            # find connected components
            cc = cv2.connectedComponents(objmask)
            cc_img = cc[1].astype(np.uint8)

            # find contours of these objects
            contours, hierarchy = cv2.findContours(cc_img,
                                                        cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)


            # Make a Kalman prediction here
            predicted_state = kalman_predict(kal_A, kal_x)

            # Use the result from contour detection as measurement
            no_object_found = True

            if contours:
                # print("This is the contour part")
                # use just the first contour to draw a rectangle
                x, y, w, h = cv2.boundingRect(contours[0])

                # Do not show very small objects
                if w > minObjectSize and h > minObjectSize:

                    no_object_found = False

                    cv2.rectangle(frame_current, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(frame_current, "Here's my candy!",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255),  1, cv2.LINE_AA)

                    # Measured position for Kalman filter
                    # Use the detection result from practical 1
                    measured_position = np.asarray([[x], [y]])

            # ***Task for you***
            # If there is no reliable detection, use the Kalman prediction as the measured position
            if no_object_found:
                measured_position = predicted_state[0:2]

            # ***Task for you***
            # Make the Kalman update here with kal_P_init = kal_P, predicted = predicted_state and measured = measured_position
            Pk, K, kal_x, kal_P = kalman_update(kal_P, predicted_state, measured_position)

            # ***Task for you***
            # Where is the estimated position?
            estimated_position = np.zeros(2)
            estimated_position = predicted_state

            estimated_position_coordinates = (int(np.round(estimated_position[0])),
                                           int(np.round(estimated_position[1])))

            cv2.circle(frame_current, estimated_position_coordinates,
                       15, (255, 0, 0), 2)

            cv2.namedWindow("Kalman Filter Tracking")
            cv2.imshow("Kalman Filter Tracking", frame_current)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()






