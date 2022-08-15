# Hyper-Stacked: Scalable and Distributed AutoML Approach for Big Data

Hyper-Stacked is implemented as a Scala package that can run on top of a distributed spark cluster. We also used MLlib, a Spark’s ML library, to make practical machine learning scalable and easy. From this library, we selected the following classifiers to be part of the algorithms that can be aprt of the ensemble built in the inner structure of Hyper-Stacked: Random Forest (RF), Gradient Boosted Trees (GBT), LinearSVC (LSVC), Logistic Regression (LR), and Naïve Bayes (NB). Besides, the meta-learner in chosen from the this portfolio of methods.

## General overview of Hyper-Stacked:
![Alt text](https://github.com/jsebanaz90/Hyper-Stacked/blob/main/Supplementary%20documents/Hyper-Stacked_workflow.JPG)

## Super Learner with Greedy k-Fold:
![Alt text](https://github.com/jsebanaz90/Hyper-Stacked/blob/main/Supplementary%20documents/Hyper-Stacked_pseudocode.JPG)
