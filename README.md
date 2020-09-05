

# Improved fraud detection pipeline using feature selction at the PySpark framework

![alt text](./img/feature_importance_pyspark.png)


## Goal
The goal of this analysis is to conduct the feature selection using PCA vs. input perturbation strategies and further enhance the model performace for fraud detection in the PySpark framework.


## Feature seelction

### Why [feature selection](https://mlwhiz.com/blog/2019/08/07/feature_selection/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com)? 

**1. Curse of dimensionality — Overfitting**

The common theme of the problems is that when the dimensionality increases, the volume of the space increases so fast that the available data become sparse. This sparsity is problematic for any method that requires statistical significance. 

In order to obtain a statistically sound and reliable result, the amount of data needed to support the result often grows exponentially with the dimensionality.

Also, organizing and searching data often relies on detecting areas where objects form groups with similar properties; in high dimensional data, however, all objects appear to be sparse and dissimilar in many ways, which prevents common data organization strategies from being efficient.

**2. Occam’s Razor**

We want our models to be simple and explainable. We lose explainability when we have a lot of features.

**3 Noise in the data**

In real applications, the data are not perfect and always noisy inherently. 


### Commonly used methods for feature selection

In summary, there are several commonly used methods to conduct feature selction in data preprocessing.

**1 Correlation or chi-square**

Chose the top-n high correlated variables or high chi-squre variables with respective to target variables.
The intuition is that if a feature is independent to the target, it will not be useful or uninformative for target classification or regression.

**2 Stepwise method**

This is a wrapper based method. The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

**3 Lasso - Penalized likelihood **

LASSO models have been used extensively in high-dimensional model selection problems, that is when the number of IVs 𝑘 by far exceeds the sample size 𝑛. 

Regression coefficients estimated by the LASSO are biased by intention, but can have smaller mean squared error (MSE) than conventional estimates. It prefer to have fewer variales with huge contribution to the target.

Because of the bias, their interpretation in explanatory or descriptive models is difficult, and confidence intervals based on resampling procedures such as the percentile bootstrap do not reach their claimed nominal level. Another problem with LASSO estimation is its dependence on the scale of the covariates

**4 PCA**

PCA is a commonly used as dimension reduction technique by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible. 

The advantages of PCA:

1. Removes Correlated Features
2. Improves Algorithm Performance
3. Reduces Overfitting
4. Improves Visualization

The things we need to consider before using PCA:

1.	Independent variables become less interpretable: 
2.	Data standardization is must before PCA: pca is affected by scale
3.	Information loss



**5 Input Perturbation**

This algorithm was introduced by [Breiman](https://en.wikipedia.org/wiki/Leo_Breiman) in his seminal paper on random forests.  Although he presented this algorithm in conjunction with random forests, it is model-independent and appropriate for any supervised learning model.  

This algorithm, known as the input perturbation algorithm, works by evaluating a trained model’s accuracy with each of the inputs individually shuffled from a data set.  Shuffling an input causes it to become useless—effectively removing it from the model. More important inputs will produce a less accurate score when they are removed by shuffling them. This process makes sense, because important features will contribute to the accuracy of the model. 


The code was coming from [here](https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/).

The variable importance can be used in other analysis.

### PCA vs. input pertubation (e.g. Gini) for Feature selection
In [previous post](https://github.com/tankwin08/PySpark_Fraud_detection_ML), I have implemented PCA method to conduct the feature selection and hyper-paramters tunning for fraud detection.
In this project, the input pertubation strategies was used to measure the feature importances.

You can have multiple ways to quantify the feature importances such as [Gini, p-values](https://www.sparkitecture.io/machine-learning/feature-importance).

The feature importance code was obtained from [here](https://gist.github.com/timlrx/1d5fdb0a43adbbe32a9336ba5c85b1b2), which is a function can be easily
incorporated into a pipeline.

Good explination of feature importances in PySpark can refer to [here](https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/).


## Data
The data was downloaded from [here](https://www.kaggle.com/ntnu-testimon/paysim1)

The explanation columns of the input data:

**One row**: 1,PAYMENT,1060.31,C429214117,1089.0,28.69,M1591654462,0.0,0.0,0,0

**Column names**: 

* step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

* type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

* amount - amount of the transaction in local currency.

* nameOrig - customer who started the transaction

* oldbalanceOrg - initial balance before the transaction

* newbalanceOrig - new balance after the transaction

* nameDest - customer who is the recipient of the transaction

* oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

* newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

* isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

* isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200,000 in a single transaction.



## Method overviews

1 Data understanding using PySpark

2 Data conversion and feature enginnering

3 Data stratified splitting and Model building

4 Pipeline (including feature selection) and cross validation

5 Model evaluation


