#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict steering angle for the car in the simulator while throttle remained constant. 

My initial work was to try recording a short trip down the initial straight road so I could analyze and visualize the image and work on the data preparation. 

My first attempt on training was to just migrate the traffic sign network from the Keras lab and see what happens. It took me a while to connect my local simulator to a model that was running in AWS and get the car moving in autonomous mode. 

I was short on time so I chose to use the Udacity dataset rather than recording my own. First few training attempts the model would produce a steering angle in the hundreds. I decided to remove all zero steering angles and this help to produce a normal output between -1 and 1. During this process I also learnt that for a continuous prediction I should remove the last activation layer which is typically used in classification networks. 

My next realization was that the network was producing a constant output regardless of which image was being used for prediction, this is also seen when it seems to settle on the same loss and validation loss within 2 epochs. I further filtered out images that were less than +-0.05 steering angles along with increasing the number of samples of larger steering angles by flipping images >0.1 and <-0.1. This produced a car that continuously turned left as I ran it in the simulator. 

Next I applied the VGG16 network and retrained it from scratch, the car was able to drive around the track, swirving from left to right. However it did manage to drive around the track when I manually reduce the speed, I'm suspecting it's got to do with the network speed or the efficiency of the model, likely the latter. The model.h5 file is over 1 gigabyte. After updating the drive.py from the udacity repo and setting the speed to 5mph the car was able to make it over the bridge and keep going, my laptop was heating up quite bad from running the simulator though. 

Trying to reduce the size of the model by using a lambda calling tensor flow as a part of the model turned out to be troublesome, the model trained ok but I couldn't load it as it encountered an undefined function in the lambda. Tried reducing the image to 80x80 but the result was worse, it turns out the default learning rate needed to be lower for the model to learn the smaller images. I set the learning rate at 0.0001 and the car was able to go around track 1 at 15mph. 

I observed that the car is still swirving a lot with this model, so I added in the left and right camera images into the training sample. This made the car a lot more stable at 15mph. 


My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.


# References
CarND forums and slack channel  
VGG paper: https://arxiv.org/pdf/1409.1556.pdf  
Nvidia paper: https://arxiv.org/pdf/1604.07316v1.pdf  
Keras documentation: https://keras.io  
Vivek's post:  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9   