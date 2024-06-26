# Project 2: The Perceptron

Team BARDO : Cindy DO, Barthélémy CHARLIER

### **Introduction**

The perceptron is a machine learning algorithm for learning a classifier of the form ;

𝑦̂= ℎ(𝒙) = 𝑠𝑖𝑔𝑛(𝒘𝑇𝒙 + 𝑏)   w is a vector of weights and b the bias

The goal of this project is to implement and train a perceptron algorithm to differentiate hand-written digits (0 from 1) .

![Fig. 1: Images to differentiate](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled.png)

Fig. 1: Images to differentiate

### Part I - Implementing a perceptron algorithm from scratch

In the [perceptron.py](http://perceptron.py/) file we implemented a ‘train’ and a ‘predict’ method following the mathematical algorithm showed in Fig. 2. Here below, we can see a snippet of the ‘train’ method.

![Capture d’écran 2023-12-14 à 23.10.31.png](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Capture_dcran_2023-12-14__23.10.31.png)

![Fig.2 The perceptron algorithm](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled%201.png)

Fig.2 The perceptron algorithm

We implemented our ‘train method’ so that the default learning rate is 0.01. We choose not to implement other method in our class in order to prioritize its efficiency.

Our ‘predict’ method predicts the labels 𝑦̂ given an input matrix x by computing: 𝑦̂= ℎ(𝒙) = 𝑠𝑖𝑔𝑛(𝒘𝑇𝒙 + 𝑏) 

### Part II - Using the Perceptron

**Loading and preparing the data**

Before starting to train our model, we first need to load and prepare our data. Indeed, ‘the load_digit( )’ method gives us a dataset of images of digits from 0 to 9, since  we only need to classify 0’s from 1’s we coded a mask to discard the unwanted data, the mask was coded using chatGPT.

![                                          Fig. 3: Before mask ](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled%202.png)

                                          Fig. 3: Before mask 

![                                                    Fig 4: With mask](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled%203.png)

                                                    Fig 4: With mask

That being done, we need to flatten the images, turning each 2-D array of grayscale values from shape (8, 8) into shape (64,). Subsequently, the entire dataset will be of shape (n_samples, n_features), where n_samples is the number of images and n_features is the total number of pixels in each image.

We last use ‘train_test_split’ from scikit-learn on the filetered dataset, to split 20% to the test set and 80% to the training set.

**Performing the k-fold  cross-validation algorithm on the training set**

In order to predict, as well as possible, the behavior of our model on unseen data, without using the testing set, we perform a 5-fold cross validation on our training set. We split the training set, using the Kfold object from scikit learn, into 5 random training and testing set and we compute the score (accuracy) of the model on each of them and then the mean score.

The computation gave us: 

Scores :[1.0, 1.0, 1.0, 1.0, 1.0]
Average Score: 1.0

The reason for our model to have that precise accuracy is that the dataset might be very simple, so much that it can be considered linearly separable. Hence, our Perceptron classifier is assured to converge.

**Results on the testing set**

Following the previous results of the cross-validation algorithm, we tested our model on the testing set, using the same hyperparameter ( learning rate of 0.01 ). We also timed our testing phase.

![                              Fig 5: Final predictions of our digit ](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled%204.png)

                              Fig 5: Final predictions of our digit 

On the testing set we have a 100% accuracy, we printed also a text report showing the main classification metrics, using the ‘classification_report method’ of scikit learn.

![Untitled](Project%202%20The%20Perceptron%209940466c5c954a5d97389181350e7a92/Untitled%205.png)

Time wise our computations are almost instant:

Our time statistics :

Predicting time: 0.001001596450805664 seconds
Training time: 0.0020029544830322266 seconds
Testing time: 0.0030045509338378906 seconds

**Comparison with Scikit-learn**

Without any surprise, Scikit-learn’s Perceptron also reached a 100% accuracy, but our model has a competitive time-efficiency. We computed the time difference between our models and they are very alike.

Time statistics of Scikit-learn  :

Predicting time: 0.0 seconds
Training time: 0.0020036697387695312 seconds
Testing - sklearn time: 0.0020036697387695312 seconds
Accuracy of sk learn's model: 100.0

Time difference (please note that far from being the case every time) 

We coded that with ChatGPT

Our custom Perceptron is 1.50 times faster than scikit-learn's Perceptron.

### Bibliography

[1]: Documentation of the dataset used: [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)

[2]: Scikit learn’s implementation of the algorithm: [https://www.kaggle.com/code/goliothzorenfredrik/handwritten-digits-recognition-perceptron](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)

[3]: StratifiedKfold: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

[4]: Classification report documentation:  [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

[5]: Le Machine learning avec Python - O’r eilly
