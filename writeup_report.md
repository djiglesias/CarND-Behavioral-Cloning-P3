# **Behavioral Cloning Project** 
## 0. Overview

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* All points from [rubric](https://review.udacity.com/#!/rubrics/432/view) are addressed

## 1. Files Submitted

Project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

## 2. Data Collecion Strategy
In order to collect a sufficient set of data to train the network two styles of driving were collected; center track (left) and recovery (right) driving. The two styles act to train the network what the idea situation is as well as how to respond to undesirable inputs where the car is straying from the track center. To provide data for the center track driving, two laps were recorded in both a clockwise and counter clockwise direction around the track. To model recovery behaviour addition video was recorded with the car returning to the middle from the edge of the track on both straight and corner track segments.

<p align="center">
 <img src="./images/center.gif">
 <img src="./images/recovery.gif">
</p>

To ensure quality of data recording was only begun once the car was up to speed and the steering was controlled via mouse input rather than keypad to avoid step respond input. This method was repeated for both tracks and recorded in separate folders to enable specific training for a single track.

## 3. Data Pre-Processing & Augmentation
### 3.1 Multiple Camera Views
Using three cameras on the car (left/center/right) increases the size of the training data set and adds multiple views to account for different perspectives of the road. Since there are three images and only one set of steering angles an offset was applied to the steering measurement and associated with the left/right images to account for the difference in view.

<p align="center">
 <img src="./images/image_left.jpg" width=250>
 <img src="./images/image_center.jpg" width=250>
 <img src="./images/image_right.jpg" width=250>
</p>


### 3.2 Mirroring Data Set
To further increase the data set size the images were flipped horizontally and appended to the data set with the negative steering measurement associated to the image. Once all the data was appended, the set was shuffled and returned using a generator.

<p align="center">
 <img src="./images/raw.png">
 <img src="./images/flip.png">
</p>

### 3.3 Keras
Using the built in methods of Keras the images were normalized and cropped to further increase the network performance. The images were left as RGB and normalized with a zero mean error between -0.5 to 0.5 for each color. The images were then cropped to remove the top and bottom rows to reduce noise and unwanted artifacts from the hood of the car and scenary unrelated to the road. By applying the normalization and cropping using Keras quite a few lines of code were saved for this project!

<p align="center">
 <img src="./images/raw.png">
 <img src="./images/crop.png">
</p>

## 4. Model Architecture

### 4.1 Solution Design Approach

- LeNet was not used.
- Nvidia was used
- Dropout was added
- pre processing

### 4.2 Final Model Architecture
The final model was the [NVIDIA Network Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with dropout layers added to prevent the network from memorizing the data set.
 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Cropped RGB Image							| 
| Normalization     | 							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 24x5x5 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 36x5x5 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 24x5x5 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 36x5x5 	|
| RELU					|												|
| Dropout          |   | 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 48x5x5 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 64x3x3 	|
| RELU					|												|
| Convolution 3x3    	| 1x1 stride, valid padding, outputs 64x3x3 	|
| RELU					|												|
| Dropout          |   | 
| Flatten          | outputs 1164  |
| Dropout          |   | 
| Fully connected		| outputs 100				  |
| Dropout          |   | 
| Fully connected		| outputs 50					|
| Fully connected		| outputs 10					|

## 5. Training the Model
Used a generator...

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

... adam optimizer
... epochs = 30
... batch size = 32

<p align="center">
 <img src="./images/training.png">
</p>
