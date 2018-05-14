# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: bar_chart_training_data.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: construction.png "Traffic Sign 1"
[image5]: speed_30.jpg "Traffic Sign 2"
[image6]: yield.jpg "Traffic Sign 3"
[image7]: right_of_way_next_intersection.jpg "Traffic Sign 4"
[image8]: left_arrow.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

You're reading it! and here is a link to my [project code](https://github.com/blown302/german-traffic-sign-detection/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the `numpy` to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the different signs. 

![alt text][image1]

### Design and Test a Model Architecture

I started out with the LeNet architecture we learned in a preview exercise. Knowing that the images at a minimum needed to be normalized I used a standard normalization tensor from tensorlfow: [per_image_standardization](https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization). I did this right in the network.

I played around with the default LeNet architecture but had to make the network a bit deeper so I added another convolutional layer and added a dropout at very end of the network. I was trying to add more dropouts at different layers but did not seem to be as effective. 

### My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3       | 1x1 stride, valid padding, relu, 2x2 maxpool, output 32x32x32|				|
| Convolution 3x3	    | 1x1 stride, valid padding, relu, 2x2 maxpool, output 32x32x64|
| Convolution 3x3		| 1x1 stride, valid padding, relu, 2x2 maxpool, output 32x32x128|
| Fully Connected       | flattened    									|
| Fully Connected		| relu, outputs 512X120 						|
| Fully Connected       | relu, outputs 120X84  						|	
| Fully Connected       | relu, dropout, outputs 84x43                  |

# Training:

To train the model, I used a batch size of 128 and 20 epocs. Used `sklearn's` shuffle to get random mini batches. Played with the learning rate but landed on a go to learning rate of `.001`. For my dropout function I stuck with `.5` keep_prob. To monitor the training process the training batch and validation batch are evaluated for accuracy. When all 20 epochs are completed we save the model to evaluate the test and internet images in the upcoming cells.

# Iterative Approach

I made many tweaks to get to the final network. First I was playing with the LeNet. Tweaks to learning rate, dropout and then extra layers and filter depth.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images I selected were pretty clear but some of them share some of the same shapes of other images in different classes.

### Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work     		| Road Work   									| 
| Speed 30     			| Speed 30										|
| Yield					| Yield											|
| Left Turn	      		| Left Turn 					 				|
| right_of_way_next_intersection| right_of_way_next_intersection		|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of `94.6%` considering the quaility of the images and the small sample size. 

All of my internet imaages were almost 100% what the answer was. In my IPython notebook there is dataframes for each of the images. I'll show the worst performing one here:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.90%       			| Road work  									| 
| 0.08%    				| Children crossing			    				|
| 0.01%					| End of all speed and passing limits   		|
| 0.01%	      			| Pedestrians					 				|
| 0.00%				    | Beware of ice/snow							|
