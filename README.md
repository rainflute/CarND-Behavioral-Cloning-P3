# **Behavioral Cloning** 



**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md for summarizing the results
* video.mp4 is the video of the car driving in autonomous mode using the trained neural network

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Following the instructor's suggestion, I used the architecture in the published article from Nvidia  

#### 2. Attempts to reduce overfitting in the model

After some experiments, I added a dropout layer with 0.5 dropout rate to reduce overfitting (model.py lines 48). 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

The training data includes images from 3 cameras mounted in the front. 

The data is collected by driving the car 2 full laps in both clockwise and counter clockwise direction in manual mode.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture from Nvidia

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I tried to add dropout layer to different positions, after few test run, I found it works better when put before the convolutional layers

After tuned the correction angle to 0.30, the loss is lower and went further in the track

The final step was to run the simulator to see how well the car was driving around track one. 

The biggest challenge came after the car crossed the bridge, there are a shape left turn and a shape right turn, where my car always drove out of track.
 
To solve the problem, I tuned the drop out rate, the correction angel and most importantly changed the training collection strategy which is explained in detail in later section.

After that, the vehicle is even more smoothly than I did in manual mode, and is able to drive autonomously around the track without leaving the road. 

#### 2. Final Model Architecture

The final model architecture in createmodel() function consisted of a convolution neural network with the following layers and layer sizes:

* Normalization layer
* Cropping layer to simplify the input data
* Drop out layer to avoid overfitting
* 3 convolutional layers with 5x5 kernel, valid padding, and 2x2 stride. The filter depth are 24, 36, 48
* 2 convolutional layers with 3x3 kernel, valid padding, and 1x1 stride with filter depth 64
* 4 fully connected layers

Here is a visualization of the architecture 

![cnn](./images/cnn-architecture.png)

#### 3. Creation of the Training Set & Training Process

I first drive the car counter clockwise as default in the simulator for 2 laps. 
After training the model, the model did a poor job when making tursn and struggles driving in the middle lane

Instead of using only the center camera, I included images from both left and right cameras and applied correction angle 0.2 as suggested by the instructor
The result was little better, drive in the middle lane, but always went out of the track at turns

Then I made a U turn and start recording driving the car clockwise. 
The result is a little bit better, but the car still cannot successfully make the shape left turn after the bridge.

Then I change the starting point near the bridge, so that I will have data of 3 shape turns in 2 laps.
I also added data generator to the code, which speed up the training process and the car finally made the shape turns after the bridge in the track.
