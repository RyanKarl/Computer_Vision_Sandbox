# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2019
# ________________________________________________________________________________
# Adam Czajka, November 2019
# Codes based on Keras documentation, more at https://keras.io/examples/mnist_cnn/

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools as it

# load database of handwritten digits (MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Let's show one example from the dataset
# image_index = 4444 # you may select anything up to 60,000
# print(y_train[image_index]) # the label is 8
# plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()

model.add(Conv2D(16, kernel_size=(5,5), strides = (1,1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))

# You can now have some fun with modifying the layers. For instance add extra convolutional layer with 32 kernels:
# model.add(Conv2D(32, kernel_size=(3,3), strides = (1,1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu')) # the number of 128 neurons selected for this hidden layer is sort of arbitrary

model.add(Dropout(0.2)) # Randomly drop 20% of connections when training (equivalent to an ensamble model learning)
model.add(Dense(10,activation='softmax')) # 10 output neurons since we have 10 classes in this task

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print('-TRAINING----------------------------')
print('Input shape:', x_train.shape)
print('Number of training images: ', x_train.shape[0])

model.fit(x=x_train,y=y_train, epochs=3)

# That's all! Let's see now how our model recognizes all test digits
print('-TESTING-----------------------------')
print('Number of test images:', x_test.shape[0])
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Print 10 example test digits with their true and predicted labels
fig, axes = plt.subplots(2, 5)
fig.tight_layout()

image_idx = np.random.randint(1,10000,(2,5))

for i, j in it.product(range(2), range(5)):
    test_image = x_test[image_idx[i,j]].reshape(1, 28, 28, 1)
    test_label = y_test[image_idx[i,j]]
    softmax_outputs = model.predict(test_image)
    pred_label = softmax_outputs.argmax()

    axes[i, j].imshow(test_image.reshape(28, 28),cmap='Greys')
    axes[i, j].set_aspect('equal', 'box')
    axes[i, j].set_title("{} / {}".format(test_label,pred_label))

plt.show()