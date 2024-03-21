# Project 3 : Ridge Regression

Team BARDO : Cindy DO, Barthélémy CHARLIER

### **Introduction**

Ridge regression is a linear regression technique that introduces a regularization term, the L2 norm. 

The goal of this project is to implement the ridge regression from scratch and to train it on the Olympics 100m dataset. 

### Part I - Implementing the ridge regression from scratch

Our RidgeReg class contains the methods `__init__`, `fit` and `predict`. The `__init__` initialises the regularization parameter α.

Then for fitting our model, we implemented the closed form solution to coefficients theta in matrix form: 

$$
\hat{\theta} = (X^T X + A)^{-1} X^T y

$$

where A is the modified identity matrix with the first element on the main diagonal (corresponding to the intercept term) is replaced with 0. This adjustment ensures that the regularization penalty does not apply to the intercept term during the ridge regression process.

Finally, after obtaining the vector of optimal coefficients for a specific α, the prediction method remains the same as in linear regression: it takes as input test data (X) and uses the adjusted coefficients (thetas) to make predictions. [2]

**Splitting strategy**

In order to avoid data leakage, we splitted the dataset by time, our training spans from 1896 to 1996 and the testing set from 2000 to 2020 ( 80/20 split using the **shuffle=False** parameter from the scikit-learn’s `train_split` method).

**Comparison with scikit-learn implementation**

To verify our results, we choose to compare our performance by using the validation with scikit learn’s Ridge model and by computing the mean square error for the two models.

![    Fig. 1: Comparaison of predicted values on the Validation Set](Project%203%20Ridge%20Regression%20f6da632751ee4da9a026af2aef4ad9ad/Capture_dcran_2024-01-25__23.22.38.png)

    Fig. 1: Comparaison of predicted values on the Validation Set

Mean Squared Error (scikit-learn): 0.07861356729361699
Mean Squared Error (custom model): 0.07861356729562581
Error Rate: 2.5553142188117653e-11
For α = 1 and value of w: [4.18105164e+01 -1.61412467e-02]

Given the error rate, we can assume that our model is correct.

We now have to find the optimal hyperparameter for our regularization.

### Part II - Hyperparameter Tuning

In order to tune our model, and taking in account the small size of the dataset we decided to use the Leave-One Out cross validation, using the `LeaveOneOut`object from scikit-learn’s `model_selection`library. Indeed, given the very small size of our data set, the leave-one-out approach is more appropriate, instead of splitting our training set into k folds, it will split it into n.  [1]

We performed the cross validation on several ranges of alphas to avoid a code too computationally intense, and plotted the mean_score vs. the alphas and calculated the optimal one.

![                                Fig. 2: Average score in function of regularization ](Project%203%20Ridge%20Regression%20f6da632751ee4da9a026af2aef4ad9ad/Capture_dcran_2024-01-25__23.29.01.png)

                                Fig. 2: Average score in function of regularization 

The optimal α value is 296.0. 

**Final plot and result**

Once the optimal α calculated, we decided to plot our model to compare it against the simple linear regression on the testing set. The mean square error of our model remains slightly below the linear

regression’s one, but on an another dataset with more multicolinearity between the points and more data, our implementation will perform better at reducing the variance as the linear regression.

![                        Fig. 3: Ridge regression comparison of results ](Project%203%20Ridge%20Regression%20f6da632751ee4da9a026af2aef4ad9ad/Capture_dcran_2024-01-25__23.34.43.png)

                        Fig. 3: Ridge regression comparison of results 

Mean Squared Error (α = 296): 0.04696892504694113
Mean Squared Error (α = 0): 0.05148365684696741
α = 296 and value of w: [ 3.68510052e+01 -1.35646736e-02]
α = 0 and value of w: [ 3.72001603e+01 -1.37439433e-02]

### Conclusion

We can see that our two models on this dataset are more or less equivalent, but our efficiency could be improved if we were to use a model with complexity, meaning adding again a little biais to reduce again  variance (such as polynomial regression for example).

### Bibliography

We used ChatGpt for the design of the cross validation for loop (as we reused our implementation of the previous project) and we used it to help us in the manipulation of data. We also used Github copilot to help with the plot of data and some of the comments. 

[1]: [https://www.baeldung.com/cs/cross-validation-k-fold-loo](https://www.baeldung.com/cs/cross-validation-k-fold-loo) 

[2]: [https://towardsdatascience.com/how-to-code-ridge-regression-from-scratch-4b3176e5837c](https://towardsdatascience.com/how-to-code-ridge-regression-from-scratch-4b3176e5837c)