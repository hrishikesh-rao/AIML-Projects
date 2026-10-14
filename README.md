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

Suppose a given class has not appeared in other category of training data, there is a chance of likelihood being equal to zero, that causes a great deal of error, hence laplace smoothening is used.

### **Steps Taken**:

-- Dataset used was SMS Spam Collection Dataset
-- The output label is encoded into 0 or 1.
-- Count Vectoriser is used to transform messages into Binary.
-- Class is used to implement NBC.
-- Laplace Smoothening is done
-- The model is implemented form scratch and evaluation metrics are computed
-- These metrics are then compared with Sklearn's model by importing it and running on the same data.
