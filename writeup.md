# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/ideatrap/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

The histogram below explores the data distribution. Some label has relative less coverage.

![Training set histogram](https://s25.postimg.org/iygel3ny7/image.png)

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the section of step 4.

As a first step, I decided to convert the images to grayscale because the most useful information is the shape in the picture rather than color. Grayscale image ignore color information

Here is an example of a traffic sign image before and after grayscaling.

![before gray](https://s25.postimg.org/rur6p1ekf/image.png)
![after gray](https://s25.postimg.org/y9q7lpla7/image.png)

As a last step, I normalized the image data so that the mean is zero. This will help to improve optimization efficiency. As it's more efficient to reach optimal, the model accuracy will also improve.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in section **Train, Validate and Test the Model**, third block of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using `train_test_split` function in `sklearn`

My final training set had 29406 number of images. My validation set and test set had 9803 and 12630 number of images.

If I have more time, I would transform the image to generate additional training set for labels are not well represented. The transformation can be done by adjusting the sharpness, blur parameters.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my model is located in the section of **Model Architecture**.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| RELU					|												|
| Convolution 3x3	    |   1x1 stride, same padding, outputs 10x10x16    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
|Flattern| outputs 400|
| Fully connected		|    outputs 84   									|
| RELU					|												|
| Fully connected		|    outputs 43   									|




#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the section of **Model Architecture**, and  **Train, Validate and Test the Model**

To train the model, I used Adam optimizer.

- The learning rate is 0.001.
- Epochs is 10.
- Batch size is 128


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the section of **Train, Validate and Test the Model**, the fifth block.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.974
* test set accuracy of 0.902


If a well known architecture was chosen:
* I chose LeNet architecture, a well known architecture.
* LeNet is straight forward, and it has fairly good performance in MNIST dataset. Traffic sign has simple geometry shape, which is similar as MNIST data.
* The final model's accuracy is above 90% for training, validation, and test set. It proves that LeNet is work reasonably well.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![sign 1](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381023?s=170667a)


![sign 2](http://media.gettyimages.com/photos/german-traffic-signs-picture-id465921879?s=170667a)


![sign3](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381295?s=170667a)


![sign4](http://bicyclegermany.com/Images/Laws/100_1607.jpg)

![sign5](http://a.rgbimg.com/cache1nHmS6/users/s/su/sundstrom/300/mifuUb0.jpg)

The second and third image may be hard to be classified because it contains secondary sign that may distort the image information

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the section of **Predict the Sign Type for Each Image**.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)     		| Speed limit (30km/h) 									|
| General caution     			| Bicycles crossing 										|
| Children crossing				| Speed limit (20km/h)											|
| Right-of-way at the next intersection	      		| Right-of-way at the next intersection|
| Road work			| Road work    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares less favorably to the accuracy on the given test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in section of **Output Top 5 Softmax Probabilities For Each Image Found on the Web**

For the first image, the model is relatively sure that this is a sign of 'Speed limit (30km/h)' (probability of 0.83), and that's the correct prediction.

For the second image, the model is only 52% certain that it's a bicycle sign. It's actually wrong. The correct answer 'General caution' is part of the top 5 predictions. The top 5 predictions are

| Probability         	|     Prediction	        					|
|:---------------------:|:-------------------------------------------:|
| .52         			| Bicycles crossing  									|
| .37     				| General caution 										|
| .06					| Traffic signals											|
| .04	      			| Go straight or right					 				|
| .01				    | Bumpy road    							|

For the third image, the model predicts it to be 'Speed limit (20km/h)' with 64% probability, and it's wrong. The correct answer 'Children crossing' is even not in the top list. The top 5 predictions are:

| Probability         	|     Prediction	        					|
|:---------------------:|:-------------------------------------------:|
| .64         			| Speed limit (20km/h) 									|
| .14     				| General caution									|
| .08					| Roundabout mandatory									|
| .03	      			| Dangerous curve to the right				 				|
| .03				    | No entry   							|

For the fourth image, the model predicts it to be Right-of-way at the next intersection with probability of almost 100%, and that's also the correct answer.


For the fifth image, the model correctly predicts it to be 'Road work' with 61% probability.
