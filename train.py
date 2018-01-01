"""Udacity Behavioural Cloning Project

Train a neural network to drive a car around a track from
images and input commands (throttle, steering, braking).
"""
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load data from file.
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
header = lines.pop(0)

# Populate measurements into array.
images = []
measurements = []
for line in lines:
	# Change path to AWS instance.
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])     #steering
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


print(len(X_train))
print(len(y_train))



# Construct basic neural network.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Model Architecture - NVIDIA.
model = Sequential()

# Normalize Images.
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

model.add(Convolution2D(6, 5, 5, activation='relu', input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dense(1, activation='softmax'))


# Layer One.
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))



# Compile & Fit Model.
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save Training Model.
model.save('model.h5')


def main():
  pass

#TODO: Left/Right camera view and steering correction factor.

if __name__ == "__main__":
  main()
