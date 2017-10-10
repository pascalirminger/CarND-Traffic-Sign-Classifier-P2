# **Build a Traffic Sign Recognition** 

[//]: # (Image References)
[image1a]: ./examples/dataset-exploratory-visualization.png "Dataset Visualization"
[image1b]: ./examples/dataset-exploratory-visualization-augmented.png "Augmented Dataset Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3a]: ./examples/augmented-data-1.png "Augmented Data - Example 1"
[image3b]: ./examples/augmented-data-2.png "Augmented Data - Example 2"
[image4]: ./examples/sign-1-speed-limit-30.png "Traffic Sign 1"
[image5]: ./examples/sign-2-speed-limit-50.png "Traffic Sign 2"
[image6]: ./examples/sign-3-no-passing.png "Traffic Sign 3"
[image7]: ./examples/sign-4-stop.png "Traffic Sign 4"
[image8]: ./examples/sign-5-road-narrows-on-the-right.png "Traffic Sign 5"
[image9]: ./examples/two-stage-convnet-architecture.png "2-stage ConvNet architecture"
<img src="examples/title-image.jpg" width="480" alt="Traffic Sign Classifier" />

## **Overview**

This repo contains the code written to complete the second project on [Udacity Self-Driving Car Nanodegree Program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) (Term 1). This project uses convolutional neural networks to classify traffic signs. The model was trained to classify traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Dataset Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs dataset:

* The size of the training set is 34799.
* The size of the validation set is 4410.
* The size of the testing set is 12630.
* The shape of a traffic sign image is (32, 32, 3); that's 32 pixels for both width and height and 3 color channels.
* The number of unique classes/labels in the dataset is 43.

Here is an exploratory visualization of the dataset. It is a bar chart showing how the data is distributed across the 43 unique traffic signs.

![Dataset Visualization][image1a]

## Design and Test a Model Architecture

As a first step, I decided to convert the images to YCrCb color space and use the y-channel. Reducing to one channel reduces the amount of input data, training the model is significantly faster. Here is an example of a traffic sign image before and after grayscaling.

![Grayscaling][image2]

As a next step, I normalized the image data for mathematical reasons. Normalized data can make the training faster and reduce the chance of getting stuck in local optima.

I decided to generate additional data because the number of images per sign varies strongly (see diagram above). ConvNet architectures have built-in invariance to small translations, scaling and rotations. When a dataset does not naturally contain those deformations, adding them synthetically will yield more robust learning to potential deformations in the test set.

The data augmentation process uses several different functions and combines them randomly to generate a jittered training dataset. Samples are randomly perturbed in position, in scale, and in rotation; they are sheared, blurred, and gamma transformed. You can find the corresponding code in the following functions:

* ```translateImage(img)```
* ```rotateImage(img)```
* ```shearImage(img)```
* ```blurImage(img)```
* ```gammaImage(img)```

The function ```processRandom(img)``` creates a new image with random adoption of the functions described earlier. Here are some examples of an original image and an augmented image:

![alt text][image3a]
![alt text][image3b]

The function ```augmentData(X, y, scale)``` enriches the dataset based on the number of images per sign.

* Signs with a lot of images (above mean) will only get a few examples added.
* Signs with only a few images (below mean) will get a lot of examples added.

The difference between the original dataset and the augmented dataset is remarkable based on the following diagram. The size of the augmented training set is 140'365.

![Augmented Dataset Visualization][image1b]

I also decided to shuffle the training set using ```shuffle``` from ```sklearn.utils``` library:

```X_train, y_train = shuffle(X_train, y_train)```

The architecture used here departs from traditional ConvNets by the type of non-linearities used, by the use of connections that skip layers, and by the use of pooling layers with different subsampling ratios for the connections that skip layers and for those that do not. As described by [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), each stage of this 2-stage ConvNet architecture is composed of a (convolutional) filter bank layer, a non-linear transform layer, and a spatial feature pooling layer.

![alt text][image9]

In traditional ConvNets, the output of the last stage is fed to a classifier. Here, the outputs of all the stages are fed to the classifier. The motivation for combining representation from multiple stages in the classifier is to provide different scales of receptive fields to the classifier. In the case of 2 stages of features, the second stage extracts “global” and invariant shapes and structures, while the first stage extracts “local” motifs with more precise details.

The final model consisted of the following layers:

| Layer           | Description                                                   | Stage         |
|:---------------:|:-------------------------------------------------------------:|:-------------:|
| Input           | 32x32x1 image with Y-channel only                             |               |
| Convolution 5x5 | input = 32x32x1, 1x1 stride, VALID padding, output = 28x28x6  |               |
| RELU            |                                                               |               |
| Max pooling     | input = 28x28x6, 2x2 stride, VALID padding, output = 14x14x6  |               |
| Convolution 5x5 | input = 14x14x6, 1x1 stride, VALID padding, output = 10x10x16 |               |
| RELU            |                                                               |               |
| Max pooling     | input = 10x10x16, 2x2 stride, VALID padding, output = 5x5x16  |               |
| Flatten         | input = 5x5x16, output = 400                                  | **1st stage** |
| Convolution 5x5 | input = 5x5x16, 1x1 stride, VALID padding, output = 1x1x400   |               |
| RELU            |                                                               |               |
| Flatten         | input = 1x1x400, output = 400                                 | **2nd stage** |
| Concat stages   | input = 400+400, output = 800                                 |               |
| Dropout         |                                                               |               |
| Fully connected | input = 800, output = 100                                     |               |
| RELU            |                                                               |               |
| Dropout         |                                                               |               |
| Fully connected | input = 100, output = 43                                      |               |

For training optimization I used the ```minimize()``` function of a ```tf.train.AdamOptimizer()``` instance using ```tf.nn.softmax_cross_entropy_with_logits()``` and ```tf.reduce_mean()``` as a loss operation. To train the model, I used the following parameters:

| Hyperparameter    | Value          |
|:-----------------:|:--------------:|
| Number of epochs  | 60             |
| Batch size        | 128            |
| Learning rate     | 0.001          |
| Keep probability  | 0.6            |

My final model results were:
* Training set accuracy of 0.976
* Validation set accuracy of 0.976
* Test set accuracy of 0.948

In a first approach, I used the ConvNet from the classroom and adjusted it to the dimensions here in the traffic sign scenario. Therefore, I increased the first layer to accept three (color-)channels instead of just one (grayscale). Also, the traffic sign classifier has 43 classes where MNIST only had 10, so I had to change that.

The main problem with that was, that the validation accuracy never went greater than 0.87. After trying different things like grayscale and normalization, which surprisingly didn't change the situation much, I ended up finding and reading the paper of [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The 2-stage architecture made a lot of sense to me, so I decided to implement this approach within the existing ConvNet from the classroom.

Improving the preprocessing as described above and tuning the hyperparameter increased the validation accuracy from .93 to .96 even more. Playing around with the hyperparameters let the model swing between overfitting and underfitting. The present configuration (see above) is the result of several testing with different values.

## Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6] 
![Traffic Sign 4][image7] ![Traffic Sign 5][image8]

The first two images might be difficult to classify because they only differ by the numbers in the sign. The third image might conflict with the other classes regarding no-passing. The fourth image with the stop sign should be the clearest within the group because of the signs shape and color distribution. The sign in the fifth image seems to be very similar to the general caution sign. It wouldn't surprise me if this image makes troubles.

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Here are the results of the prediction:

| Image                     | Prediction                        | Result  |
|:-------------------------:|:---------------------------------:|:-------:|
| Speed limit (30km/h)      | Speed limit (30km/h)              | True    |
| Speed limit (50km/h)      | Speed limit (50km/h)              | True    |
| No passing                | No passing                        | True    |
| Stop                      | Stop                              | True    |
| Road narrows on the right | General caution                   | False   |

For the first image, the model is absolutely sure that this is a ```'Speed limit (30km/h)'``` sign (probability of 1.0), and the image does contain a ```'Speed limit (30km/h)'``` sign. The top five soft max probabilities were:

| Probability  | Prediction                                |
|-------------:|:------------------------------------------|
| 1.00         | Speed limit (30km/h)                      |
|  .00         | Speed limit (20km/h)                      |
|  .00         | Speed limit (70km/h)                      |
|  .00         | Speed limit (50km/h)                      |
|  .00         | Speed limit (80km/h)                      |

For the second image the model is absolutely sure that this is a ```'Speed limit (50km/h)'``` sign (probability of 1.0), and the image does contain a ```'Speed limit (50km/h)'``` sign. The top five soft max probabilities were:

| Probability  | Prediction                                |
|:------------:|:-----------------------------------------:|
| 1.00         | Speed limit (50km/h)                      |
|  .00         | Speed limit (60km/h)                      |
|  .00         | Speed limit (30km/h)                      |
|  .00         | Speed limit (80km/h)                      |
|  .00         | Speed limit (120km/h)                     |

For the third image the model is absolutely sure that this is a ```'No passing'``` sign (probability of 1.0), and the image does contain a ```'No passing'``` sign. The top five soft max probabilities were:

| Probability  | Prediction                                |
|:------------:|:-----------------------------------------:|
| 1.00         | No passing                                |
|  .00         | End of no passing                         |
|  .00         | Speed limit (30km/h)                      |
|  .00         | End of all speed and passing limits       |
|  .00         | Vehicles over 3.5 metric tons prohibited  |

For the forth image the model is absolutely sure that this is a ```'Stop'``` sign (probability of 1.0), and the image does contain a ```'Stop'``` sign. The top five soft max probabilities were:

| Probability  | Prediction                                |
|:------------:|:-----------------------------------------:|
| 1.00         | Stop                                      |
|  .00         | Keep right                                |
|  .00         | Speed limit (60km/h)                      |
|  .00         | Speed limit (50km/h)                      |
|  .00         | Ahead only                                |

For the fifth image the ConvNet predicts ```'General caution'```, which is wrong. On of the problems with the correct ```'Road narrows on the right'``` is the lack of a greater presence in the training dataset. Since ```'General caution'``` and ```'Road narrows on the right'``` are slightly similar traffic sings, it is no surprise that ```'General caution'``` is the ConvNets best guess instead. We have to keep an eye on being ```'Road narrows on the right'``` its second best guess. The top five soft max probabilities were:

| Probability  | Prediction                                |
|:------------:|:-----------------------------------------:|
|  .67         | General caution                           |
|  .33         | Road narrows on the right                 |
|  .00         | Traffic signals                           |
|  .00         | Pedestrians                               |
|  .00         | Road work                                 |
