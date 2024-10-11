# Random Forest Regression Model

This repository contains a Random Forest Regression model implemented in Python. The model is trained to predict prime numbers based on rank and interval data.

## Introduction

The goal of this project is to predict prime numbers using a Random Forest Regression model. The dataset consists of three columns: `rank`, `num`, and `interval`. The `rank` and `interval` are used as features to predict the `num`, which represents a prime number.

## Mathematical Background

### Coefficient of Determination (R²)

The R² score is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable. It is calculated as:

\[ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \]

where \( \bar{y} \) is the mean of the actual values.

## Random Forest Algorithm

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees. It helps to improve the predictive accuracy and control over-fitting.

### Steps Involved

1. **Bootstrap Sampling**: Random subsets of the training data are created with replacement.
2. **Decision Trees**: Multiple decision trees are trained on these subsets.
3. **Aggregation**: The predictions from all the trees are averaged to produce the final prediction.

### Advantages

- **Reduces Overfitting**: By averaging multiple trees, the model reduces the risk of overfitting.
- **Handles Large Datasets**: Efficiently handles large datasets with higher dimensionality.
- **Feature Importance**: Provides insights into feature importance.

## Plots

The model generates the following plots to visualize the results:

1. **Scatter Plot**: Shows the relationship between true values and predictions.
    
![](https://github.com/serize02/random_forest-primes-model/blob/main/plots/plot_2024-10-11%2018-09-05_0.png)

2. **Line Plot**: Compares actual and predicted values over sample indices.

![](https://github.com/serize02/random_forest-primes-model/blob/main/plots/plot_2024-10-11%2018-09-05_1.png)

3. **Error Histogram**: Displays the distribution of prediction errors.

![](https://github.com/serize02/random_forest-primes-model/blob/main/plots/plot_2024-10-11%2018-09-05_2.png)

