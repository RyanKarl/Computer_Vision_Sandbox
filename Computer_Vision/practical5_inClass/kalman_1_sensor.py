# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ______________________________________________________________________
# Jin Huang, Aakash Bansal, Adam Czajka, September 2018 -  November 2019

# import useful libraries
import cv2
import numpy as np
from skimage import measure
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

# Parameters and paths
video_path = "MarkerCap_small.mp4"

# Note 1: for high FPS (frame per second), exceeding 25, consider skipping
# every k frames (i.e., make "frameStep" larger than 1):
frame_step = 3

h_lower, h_upper = 63, 75
s_lower, s_upper = 25, 255
v_lower, v_upper = 51, 230

# HSV threshold for the object
#h_lower, h_upper = 20, 26
#s_lower, s_upper = 130, 190
#v_lower, v_upper = 130, 255

# Kalman filter parameters
kal_x = np.matrix([0, 0, 0, 0]) # state, in this case rigged as [x y dx dy]
kal_P = np.identity(4) # filter confidence (state covariance)
kal_C = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])

# Transition matrix
dt = 1
kal_A = np.matrix([[1, 0, dt, 0],
                    [0, 1,  0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

# Noise estimates
kal_R = 0.1 * np.identity(2) # measurement noise
kal_Q = 0.1 * np.identity(4) # process noise


# Definition of "Lego bricks" for this practical:

def object_detect(frame):
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Select the proper hsv value
    mask = np.zeros((image_hsv.shape[0], image_hsv.shape[1]))

    for i in range(image_hsv.shape[0]):
        for j in range(image_hsv.shape[1]):
            pixel_h = image_hsv[i, j, 0]
            pixel_s = image_hsv[i, j, 1]
            pixel_v = image_hsv[i, j, 2]

            if ((pixel_h >= h_lower) and (pixel_h <= h_upper) and
                (pixel_s >= s_lower) and (pixel_s <= s_upper) and
                (pixel_v >= v_lower) and (pixel_v <= v_upper)):

                mask[i, j] = 1

    labels = measure.label(mask, 4)
    features = measure.regionprops(labels)

    if (features != None):
        # Find the index of the biggest blob
        all_area_size = []

        for i in range(len(features)):
            area_size = features[i].area
            all_area_size.append(area_size)

        # Find the index of the biggest area
        index_of_biggest_blob = np.argmax(all_area_size)

        # Get the coordinates of the centroid
        x = features[index_of_biggest_blob].centroid[0]
        y = features[index_of_biggest_blob].centroid[1]

        # position = [x, y]
        position = [[x], [y]]

        return position

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

    # Load the video file and count number of frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("There are %d frames in the video." % frame_count)

    nb_frame = 0
    print("Tracking started...")

    while (cap.isOpened() and (nb_frame < frame_count - frame_step)):
        for i in range(frame_step):
            ret, frame_current = cap.read()
            nb_frame += 1

        # Measurement, prediction and update

        # ***Task for you***
        # Make a Kalman prediction here:
            #print(frame_current)
            #print(ret)
            predicted_state = kalman_predict(kal_A, kal_x)

        # Object detection (measurement)
        try:
            measured_position = object_detect(frame_current)

            center_coordinates = (int(np.round(measured_position[1])),
                                  int(np.round(measured_position[0])))

            cv2.circle(frame_current, center_coordinates, 10, (0, 255, 0), 2)

        except ValueError:

            # ***Task for you***
            # If we do not have a measurement for this frame, let's use the predicted state:
            measured_position = predicted_state[0:2]

        # ***Task for you***
        # Make the Kalman update here with kal_P_init = kal_P, predicted = predicted_state and measured = measured_position
        Pk, K, kal_x, kal_P = kalman_update(kal_P, predicted_state, measured_position)

        # ***Task for you***
        # Where is the estimated position?
        estimated_position = predicted_state
        # estimated_position = np.zeros(2)

        center_coordinates_estimate = (int(np.round(estimated_position[1])),
                                        int(np.round(estimated_position[0])))

        cv2.circle(frame_current, center_coordinates_estimate, 10, (255, 0, 0), 2)
        cv2.imshow("Tracking result", frame_current)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Tracking finished.")
    cap.release()
    cv2.destroyAllWindows()
