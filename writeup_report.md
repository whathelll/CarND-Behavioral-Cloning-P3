# **Behavioral Cloning**
# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[steering-distribution]: ./examples/udacity-data-steering-distribution.png "Steering distribution"
[before-crop]: ./examples/before-crop.png "Before cropping"
[after-crop]: ./examples/after-crop.png "After cropping"
[center-normal]: ./examples/center-normal.png "Center camera"
[center-flipped]: ./examples/center-flipped.png "Center camera flipped left and right"
[left-camera]: ./examples/left-camera.png "Left camera"
[right-camera]: ./examples/right-camera.png "Right camera"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 256 (model.py lines 55-91)

The model includes RELU layers to introduce nonlinearity (code line 55-91), and the data is normalized in the model using a Keras lambda layer (code line 57).  

I used a modified VGG architecture, see below for the breakdown.  



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 61, 65, 69, 74, 79).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 233). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the Udacity data (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)  

I wanted to push it as far as possible without adding additional training data. I wanted to see how far augmentation can go.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

This is going to be more of a diary on how I arrived at the final architecture rather than a strategy :)

My initial work was to try recording a short trip down the initial straight road so I could analyze and visualize the image and work on the data preparation.

My first attempt on training was to just migrate the traffic sign network from the Keras lab and see what happens. It took me a while to connect my local simulator to a model that was running in AWS and get the car moving in autonomous mode.

I was short on time so rather than mucking around with the simulator I chose to use the Udacity dataset rather than recording my own. First few training attempts the model would produce a steering angle in the hundreds. I decided to remove all zero steering angles and this help to produce a normal output between -1 and 1. During this process I also learnt that for a continuous prediction of steering I need to remove the last activation layer which is typically used in classification networks otherwise it becomes a % chance.

My next realization was that the network was producing a constant output regardless of which image was being used for prediction, this is also seen when it seems to settle on the same loss and validation loss within 2 epochs. I further filtered out images that were less than +-0.05 steering angles along with increasing the number of samples of larger steering angles by flipping images >0.1 and <-0.1. This produced a car that continuously turned left as I ran it in the simulator.

Next I applied the VGG16 network, added layers in front and back and retrained it from scratch, the car was able to drive around the track, swirving from left to right. However it did manage to drive around the track when I manually reduce the speed, I'm suspecting it's got to do with the network speed or the efficiency of the model, likely the latter. The model.h5 file is over 1 gigabyte. After updating the drive.py from the udacity repo and setting the speed to 5mph the car was able to make it over the bridge and keep going, my laptop was heating up quite bad from running the simulator though.

Trying to reduce the size of the model by using a lambda calling tensor flow as a part of the model turned out to be troublesome, the model trained ok but I couldn't load it as it encountered an undefined function in the lambda. Tried reducing the image to 80x80 but the result was worse, it seems the learning rate needed to be lower for the model to learn the smaller images. I set the learning rate at 0.0001 and the car was able to go around track 1 at 15mph.

I observed that the car is still swirving a lot with this model, so I added in the left and right camera images into the training sample. This made the car a lot more stable at 15mph. However, it was not able to climb hills and take sharp turns properly for the 2nd track.

After doing some reading around and chatting to people on slack. I found Vivek's blog post interesting, and come to see how the generator can really make it convenient to produce more varied data for testing without having to always go through 10k images. I did a simple generator that just randomly picked images (from a sample that has already filtered out all the near 0 images) for training, this was also able to drive around the track in slow speed.

After playing around a little bit. I came to the theory that it's the data that makes the difference, so I modified the sampling method to create a flatter normal distribution per batch. However this didn't turn out to work as well as I had expected. I then reverted to the original generator and tried the Nvidia model, that also didn't provide much improvement.

After removing some of Vivek's ideas (transformation and random brightness), the car was more stable again and the training error was lower. I then modified the activations from relu to elu that provided a slight improvement. After removing the transform that attempts to mimic sharp turns the result was a lot better.

Tested training + validation + driving at 64x64 vs 80x320, higher resolution images don't seem to make much of a difference, infact it was slightly worse at the higher resolution which was interesting.


After a week of hardly any improvement to my results, I came to the realization that my CNN layers did not have activations, which means it's a straight linear function. So happy yet so annoyed. I'm now  yielding much better results after putting activation into the convolutional layers. Threw away some of the augmentation stuff since it didn't seem to make a difference for track 1. The car can now drive at 20mph nicely on track 1, can also do it comfortably in reverse.  

Still can't do 30mph yet, I suspect I will need new data for this as the steering angle will need to be lower during straights  and stronger during corners at higher speed. I'm also suspecting that my setup of having my model on AWS and running the simulator on my local inherently carries a lag effect where the car cannot respond fast enough. Other things I could explore is to create a graph with 2 inputs (picture and current speed) which can output both the steering and the throttle. These are theories which I don't have enough time to investigate or confirm as I need to move on to P4.

I did not explore track 2 too much other than try it out here and there, I had hoped the model could generalize to track 2 without much work but after some brief exploration I believe I'd have to record data there. Vivek's post regarding augmentation appears to be only relevant to the old track 2 which was similar to track 1 but with shadows etc.   

It's been a really fun and challenging project.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-91) consisted of a convolution neural network with the following layers and layer sizes ...

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 32)    128         lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 64, 32)    9248        convolution2d_1[0][0]            
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 32, 32, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 32, 32)    0           averagepooling2d_1[0][0]         
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 32, 32, 64)    18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 32, 32, 64)    36928       convolution2d_3[0][0]            
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 16, 16, 64)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 16, 16, 64)    0           averagepooling2d_2[0][0]         
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 16, 16, 128)   73856       dropout_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 16, 16, 128)   147584      convolution2d_5[0][0]            
____________________________________________________________________________________________________
averagepooling2d_3 (AveragePooli (None, 8, 8, 128)     0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 8, 8, 128)     0           averagepooling2d_3[0][0]         
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 8, 8, 256)     295168      dropout_3[0][0]                  
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 8, 8, 256)     590080      convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 8, 8, 256)     590080      convolution2d_8[0][0]            
____________________________________________________________________________________________________
averagepooling2d_4 (AveragePooli (None, 4, 4, 256)     0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 4, 4, 256)     0           averagepooling2d_4[0][0]         
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 4, 4, 256)     590080      dropout_4[0][0]                  
____________________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 4, 4, 256)     590080      convolution2d_10[0][0]           
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 4, 4, 256)     590080      convolution2d_11[0][0]           
____________________________________________________________________________________________________
averagepooling2d_5 (AveragePooli (None, 2, 2, 256)     0           convolution2d_12[0][0]           
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 2, 2, 256)     0           averagepooling2d_5[0][0]         
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1024)          0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1024)          1049600     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1024)          1049600     dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 512)           524800      dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 256)           131328      dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 128)           32896       dense_4[0][0]                    
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 64)            8256        dense_5[0][0]                    
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 32)            2080        dense_6[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 1)             33          dense_7[0][0]                    
====================================================================================================
Total params: 6,330,401
Trainable params: 6,330,401
Non-trainable params: 0


```

#### 3. Creation of the Training Set & Training Process

I simply used the udacity data to see what I could do with it.

In examining the data the most obvious point is that the data is skewed heavily towards centre driving.
![alt text][steering-distribution]

One of the first things I did was to simply filter out the low zero angles
```
  df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
```
While I suspect this placed an undesirable effect placing a stronger emphasis on non straight driving, when I did try a different approach to sampling method to create a flat distribution out of the udacity data it turned out to be ineffective, that was before I realized I didn't have activations in my convolution layers so it's probably worthwhile exploring that again.


To even out the data and ensuring that there's no left turn bias I did image flipping
```
flip = random.randint(0, 1)
if flip == 1:
    img = np.fliplr(img)
    steering = -steering
```
![alt text][center-normal]
![alt text][center-flipped]


I also made use of the left and right cameras to increase data samples as well as help to reduce swirving of my car during straights.

```
img = None
steering = None
camera = ['left', 'center', 'right']
adjustment_value = 0.1
steering_adjustment = [adjustment_value, 0, -adjustment_value]
camera_selection = random.randint(0, 2)

steering = row['steering'] + steering_adjustment[camera_selection]

img = mpimg.imread(row[camera[camera_selection]])
```
![alt text][left-camera]
![alt text][center-normal]
![alt text][right-camera]


After reading the Nvidia paper I converted my image from RGB to YUV which seemed to have a small positive improvement. I suspect the grayscale channel helps.
```
img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
```

I tried random brightness adjustment and image transformations as used by Vivek in his post, in the end I discarded it as it didn't seem to have a positive impact on the final model. I wonder if this is a case of overfitting.

I used a generator for training which will generate randomly provided samples which is the outcome of all of the above mentioned augmentation and preprocessing.

**My training parameters:**  
Learning rate = 0.00001
Optimizer = adam  
Error measurement = Mean Squared Error  
Batch size = 256
Epochs = 30
Samples per Epoch = 256*40

For validation I used a separate generator which randomly samples the original data.

# Lessons learnt
1. Activations are important and Keras by default use a simple f(x) -> x
2. Data quality is very important. Should have tried recording my own driving for more data points.
3. Model complexity and impacts, my first VGG model was 1.2GB and training took a long time, that was before resizing my input. In the end I was still able to achieve good results with a simpler and more shallow convolution layers.
4. Data augmentation techniques
5. The difference between classification and continuous prediction
6. Learnt a bit more about Elu, dropouts and the Nvidia approach
7. Much more comfortable with Keras compared to before

# References
CarND forums and slack channel  
VGG paper: https://arxiv.org/pdf/1409.1556.pdf  
Nvidia paper: https://arxiv.org/pdf/1604.07316v1.pdf  
Keras documentation: https://keras.io  
Vivek's post:  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9  
Simultor: https://github.com/udacity/self-driving-car-sim
