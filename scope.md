# ST2MML
8 sessions, 3 to 4h each
Language: English

## Schedule
- 07/02/2025 ven 09:10 13:40 04:30
- 14/02/2025 ven 09:10 12:30 03:20
- 21/02/2025 ven 09:10 12:30 03:20
- 07/03/2025 ven 09:10 12:30 03:20
- 11/03/2025 mar 08:00 12:30 04:30
- 14/03/2025 ven 09:10 12:30 03:20
- 17/03/2025 lun 13:50 18:20 04:30
- 18/03/2025 mar 13:50 18:20 04:30

## Launch
**Format**: slides

**Content**:
- Welcome
- Scope: 3 types of models: Linear, Tree based, Neural networks
- Using LLMs in the course
- Evaluation
- Student get to know, questionnaire

## Introduction to ML
**Format**: markdown, pdf

**Content**:
- Cycle of a data science project
- What is the concept of teaching a computer using data: what is machine learning, the sqrt(2) example, vocabulary: model, inference, training
- Applications of Machine Learning: nature of data, multimodality
- Stats vs Machine Learning: explanation vs inference
  - Illustrate with questions: find factors influencing vs predicting; understanding vs inference; ex users click on website
  - Back box: interpretability, recruitment, credit
- Supervised vs non-supervised, ground truth data
- Regression, Classification, and Clustering
- Choosing the right Metric, scoring
- The importance of Benchmarking
- It's all about the data: UCI ML repo (1978), tabular
- Kaggle competitions: titanic, boston housing

## Hands on with sklearn
**Goal**: work on iris with a classifier

**Format**: slides

**Content**:
- Structure of a notebook, example on iris
  - Load data (always use pandas and csv)
  - Explore data: min, avg, percentiles, missing, categories, correlation
  - Transform data: clean, encode, engineer
  - Choose sklearn model (start with logistic regression and linear regression to get a benchmark)
  - Build X and y (design matrix and target)
  - Train the model (fit)
  - Evaluate the score
- Then get on colab
- And start coding

## Project
**Description**:
- Find a dataset on UCI ML repo, explore data, choose a target, train multiple models
- Fine tuning strategy
- Demo perf
- Includes github or notebook
- Presentation, last session
- Fill spreadsheet
- Group work
- The more complex the better, challenge yourselves

## Linear regression statistic
**Title**: Linear Regression - statistic - statsmodel

**Content**:
- Simple Example: weight ~ age, height, ... ; advertising
- Applications of linear regression
- Linearly separable data (y ~ x^2 => not linearly separable)
- LR as statistic modeling: understanding the dynamic of variables
- LR as machine learning: training a model for inference
- Interpretation of statistical LR (statsmodl): R^2, AIC, ...
- Normalization
- Hypothesis: indep of resiuduals, heteroskedasticity, ...
- Feature engineering and polynomial regression
- Explainability: not a black box model, useful for credit rating and other applications

**Demo**: advertising
**Practice**: children, stats model
**Reference**: https://www.statlearning.com/

## Linear regression machine learning
**Title**: Linear Regression - machine learning - sklearn

**Context**:
- Sklearn overview: model, fit, predict, score, random state
- Keep some data aside for validation
- Design matrix, target variable
- Bias variance, overfitting
- K-fold cross validation
- Model performance evaluation (overfitting, bias-variance, crossfolding, ...)
- Regression metrics RMSE, MAE, MAPE, etc, absolute and relative

**Demo**: children weights; advertising; other

**Practice**: other; maybe artificial

## Pandas dataframes and data engineering
**Title**: feature engineering with pandas

**Content**:
- Exploring: describe, info and correlation
- Clean data:
  - Nulls and missing values
  - Encoding categorical data
  - Outliers
- Python:
  - Avoiding loops with comprehensions
  - Creating new features with lambdas
- Merging, concatenating
- Looping through data, the list of dict to panda conversion

**Demo**: some noisy dataset (trees?)
**Practice**: some other dataset, other transformations, ADEME

## Logistic regression
**Title**: LR for classification

**Content**:
- How it works: logit function f(y) = 1 / (1 + e^(-y)) + probability interpretation
- Classification metrics, confusion matrix, Recall, AUC, etc ...
- Importance of threshold
- Probability histogram
- Encoding categorical data, curse of dimension
- Multiclass classification
- No free lunch

**Demo**: iris, penguins, breast cancer
**Practice**: titanic on kaggle, cancer dataset

## Stochastic Gradient
**Title**: the building block of today's AI

**Content**:
- Sqrt(2): Loss function, error, learning rate
- L1, L2
- What converges better?
- Gradient descent
- Tuning the learning rate
- Tuning the batch size
- Overfitting
- Regularization: L1, L2, ..., lasso

**Demo**: titanic
**Practice**: space titanic

## Cost Function and models
**Title**: going from one model to the other by changing the loss function

**Content**:
- How to choose a model
- Generalized Linear Model (GLM)
- Main loss functions
- Why the loss function dictates the model
- Examples

**Demo**: find a model that uses different loss functions

## Perceptron
**Title**: from stochastic gradient to neural networks

**Content**:
- The perceptron, single cell, with logit activation function and sign based updates
- Activation functions
- Backpropagation
- Multi layer perceptron
- Tensorflow, pytorch, jax
- Image classification

**Demo**: cats and dogs
**Practice**: spectrograms of sound?

## Unsupervised
**Title**: no ground truth

**Content**:
- Dimension reduction, PCA
- K-means, Nearest Neighbors, variants
- Silhouette score

## Trees
**Title**: decision trees

**Content**:
- Depth, leaves
- Gini impurity, entropy, and information gain
- Left unconstrained the tree always overfit

**Demo**: trees on iris
**Practice**: trees on simple classification

## Random Forest
**Title**: bagging and boosting decision trees

**Content**:
- Weak learner
- Bagging
- Impact on overfitting and biais

**Demo**: manual averaging of trees; bagging linear regression
**Practice**: tune a random forest

## XGBoost
**Title**: boosting decision trees

**Content**:
- Boosting
- Different flavors of XGBoost
- Comparison of meta parameters

**Practice**: tune a XGBoost

## Prerequisites
- Linear algebra
- Probability calculus
- Matrix analysis
- Programming

## Learning Outcomes
At the end of this course, students will be able to:
- Understand machine learning and its applications
- Learn the basic algorithms of Machine Learning
- Learn to use Python: a powerful programming language to apply ML
- Learn to implement several Machine Learning algorithms (from scratch) using Python
- Learn to build machine learning systems (ML Model) with concrete examples
- Learn to analyze the results of an ML model, to deduce conclusions, and to improve it
- Describe mathematically the problems to be solved and solve them algorithmically