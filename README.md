# Advanced ML and Intro to Deep Learning
--- 
# Task 1: Naive Bayesian Classifier

## Aim: 

Understand and implement the NBC from scratch.

## Learnings:

### ***What is Bayes Theorem***:

Bayes theorem states that the probability of event A occurring given that event B has occurred is equal to the probability of event B occurring given that A has occurred, multiplied by the probability of A, divided by the probability of B.

In formal terms, it is a mathematical theorem that describes how to update the probability of a hypothesis or event when new evidence becomes available, using the prior probability of the hypothesis and the likelihood of observing that evidence.

![Screenshot](https://github.com/hrishikesh-rao/task/blob/main/Screenshot%202026-10-10%20221954.png?raw=true)

P(A\B) : Posterior probability: probability of event A given that event B has occurred.

P(B\A) : Likelihood: probability of observing B if A is true.

P(A) : Prior probability: probability of A before observing B.

P(B) Evidence: overall probability of observing B.

Based on this Naive Bayesian Classifer is implemented.

### **Naive Bayesian Classifer**

Naive Bayes Classifier is a probabilistic machine learning classification algorithm based on Bayes theorem that predicts the class of a given data sample by calculating the posterior probability of each possible class, under the assumption that all features are conditionally independent given the class label.

Naive assumption: It assumes that features are conditionally independent of one another when the label is known.
The assumption is often not perfectly true in real data, but Naive Bayes can still work well for text classification.

#### **Laplace Smoothening** :

Suppose a given class has not appeared in other category of training data, there is a chance of likelihood being equal to zero, that causes a great deal of error, hence laplace smoothening is used, i.e. a small value is added such that the entire probability does not turn zero.

### **Steps Taken**:

- Dataset used was SMS Spam Collection Dataset.
- The output label is encoded into 0 or 1.
- CountVectorizer is used to transform messages into binary vectors.
- Class is used to implement NBC.
- Laplace smoothing is applied.
- The model is implemented from scratch, and evaluation metrics are computed.
- These metrics are then compared with Scikit-learn's model using the same data.
-- The model is implemented form scratch and evaluation metrics are computed
-- These metrics are then compared with Sklearn's model by importing it and running on the same data.

  ---

  # Task 2: Ensemble Learning and Tree Based Models

## Aim:

To understand tree based models and how combining them improves overall efficiency.

## Learnings:

### **Random Forest**:

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make a final prediction. It uses random sampling of training data and random selection of features to introduce diversity among the trees, improving prediction stability and reducing overfitting compared with a single decision tree in many cases.

A random forest is much more efficient than a simple decision tree as they employ two main things i.e. bootstrap sampling and random feature selection

**Bootstrap sampling**: Each tree is trained on a different bootstrap sample of the original training dataset.Bootstrap sampling means randomly selecting training examples with replacement.

Consider Training sample as {A,B,C,D,E}

Tree 1 after Bootstrap sampling : {A,B,B,E,D}\
Tree 2 after Bootstrap sampling : {C,B,A,E,B}
and so on

**Random feature selection**: In a normal decision tree, the root node is determined by the gini entropy if done normally or the id3 method. The features are assigned in the order of correct measurement of samples, similarly the next node is chosen.

But here there is no regard given to that entropy score. At each split in a tree, Random Forest considers only a randomly selected subset of the available features rather than necessarily considering all features.

This encourages trees to learn different patterns instead of repeatedly relying on the same dominant features.

Since Classification is done, a voting between the many trees is done, and the choice with the highest vote is said to prevail.

### Gradient Boosting Machine (GBM) 

Gradient Boosting Machine (GBM) is an ensemble machine learning algorithm that builds decision trees sequentially, where each new tree is trained to improve the predictions of the existing model by reducing a specified loss function.

We start by considering the target, if its numeric we take the average of the target and then compute residuals (which is actual value minus the mean), very similar to how error is computed in linear regression.

Next a tree is implemented based on the features wherein on the leaves, the residuals are accounted for.The residuals inside a leaf , their mean is taken.

The mean residual values become the new residuals and they are used to compute the target variable.
Like linear regression there is the use of Learning rate, so as to slow down the magnitude of prediction.

This process is repeated for couple of times to obtain suitable predictions

The mathematical model is stated below

![](https://github.com/hrishikesh-rao/task/blob/main/Screenshot%202026-10-11%20190459.png?raw=true)

For one iteration 

![](https://github.com/hrishikesh-rao/task/blob/main/Screenshot%202026-10-11%20190943.png?raw=true)

The process continues until the chosen number of trees is reached or a stopping criterion is satisfied.

Here the Loss Function is MSE.\
It is called gradient boosting as the gradient of the loss function is taken into account.

The gradient is also multiplied with the negative sign.

Boosting means since there are multiple trees all which have a very small learning rate, combining them lead to a improvement in the overall model.
That is combining a lot of weak learners to do the given task.


### XG Boost:

XGBoost is an optimized gradient-boosting machine learning system that builds an additive model of decision trees by minimizing a loss function while incorporating regularization to control model complexity.

A major feature of XGBoost is that its tree-building objective uses both the first and second derivatives of the loss function.

A major feature of XGBoost is that its tree-building objective uses both the first and second derivatives of the loss function.
- Gradient (g_i): Describes the direction and strength of the loss change.
- Hessian (h_i): Describes the second derivative, providing information about the curvature of the loss.

A main feature of XG Boost is regularization

Regularization means that mkaing sure theres a valid comparison while trees are being compared. Example: A tree with 5 leaves and 100 leaves cannot be compared, if they receive similar accuracy whilst training

Objective=Training Loss + Regularization

![](https://github.com/hrishikesh-rao/task/commit/821223ef96b8401a0bcba261bcd3b1bfb1c62bee?raw=true)


### Steps Taken:

- The titanic survival dataset is used to demonstrate ensemble learning.
- Data is preprocessed 
- Label encoder is used to encode gender, male:0,female:1 and S/C/Q as 0/1/2
- The models are imported and trained on the data
- The evaluation metrics for the models are computed.
- Finally to show ensemble, a voting classifier between the models is implemented and metrics are compared.

---

# Task 3: Hyperparameter Tuning

## Aim:

To understand and implement hyperparameter tuning and their influence on metrics.

## Learnings:

### Hyperparameters vs Parameters

Parameters are learned by the model during training (e.g., thresholds in a decision tree). Hyperparameters are set before training and control how the model learns.

### Grid Search

Grid Search exhaustively tries every combination of hyperparameter values defined in a parameter grid. Each combination is evaluated using cross-validation and the best scoring combination is selected.

### Cross-Validation

Instead of evaluating on a single train-test split, cross-validation (cv=5) splits the training data into 5 folds, trains on 4 and validates on 1, rotating through all folds. This gives a more reliable performance estimate and prevents overfitting to one particular split.

## Steps Taken:

--Iris dataset is loaded from sklearn and converted to a DataFrame
--Random Forest classifier is trained with default parameters as the baseline
 n_estimators — number of trees in the forest
 max_depth — how deep each tree can grow
 min_samples_split — minimum samples required to split a node
--Grid Search is run over n_estimators, max_depth, and min_samples_split
--Best configuration found: max_depth=None, min_samples_split=2, n_estimators=200 with a CV accuracy of 0.95
--Baseline and tuned test accuracy were both 1.0 — Iris is too simple a dataset to reflect tuning impact on the test set, though cross-validation scores showed variation (0.941 → 0.95)

--Iris is quite a simple dataset to observe Hyper Parameter tuning hence both the baseline and tuned test accuracy are 1.
--If a more complex dataset were to be chosen, then a marginal difference could be ibserved.
