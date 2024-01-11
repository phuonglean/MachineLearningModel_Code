# MachineLearningModel_Code
Standard code for Machine Learning models and will be updated every new model I have learned on Machine Learning Specialization on Coursera.

Last Updated Date: Jan 11th, 2024

I have completed Supervised Learning and Advanced Learning algorithms courses which cover:

**1. Linear Regression** (with Gradient Descent, Cost function computation, Feature Scaling, SGDRegression)

**2. Logistic Regression** (Batch Gradient Descent, Cost function computation)

**3. Neural Network, Decision Tree, Random Forest, XGBoost** (with Neural Networks, matrix, activation function [Sigmoid, ReLu, Linear, Softmax], roundoff error, convolutional neural network, back propagation, Bias & variance, Decision Tree, Entropy measurement, Tree ensemble & Random Forest, XGBoost for Random Forest)

Besides, I have also find a dataset on Kaggle to practice learning points described above:

Case study brief: Customer profiles dataset at one bank comprises values of Credit Score, Geography, Gender, Age, Tenure, Balancing, Number of products, Credit Card possession, Active Member, Estimated Salary, Excited. Based on these columns, a bank wants to predict whether the customers will churn.

First Approach: To start with simple method, I use ANN model to predict the probability of churn with 6 steps below:
- Step 1: Import libraries data - I use sklearn libraries and tensorflow package whereas the dataset is on Kaggle named Bank Customer Churn Dataset
- Step 2: Explore data - I use visualization to demonstrate distribution, skewness and correlations of variables
- Step 3: Preprocess data - I use one-hot encoding to treat categorical variables. Then splitting dataset into train and test set
- Step 4: Feature scaling - Because each features differ from each other scales, I use standard scaler which subtract value on each row's feature by mean and then divide it by standard deviation as for 2 reasons: first is distributions of all features are not normally distributed; second is each feature has different value range which increase time to reach optimal global minimum of loss function
- Step 5: Build model - For ANN: I develop simple model with 3 layers: first layer is input layer with 13 units, activation function is RELU; second layer is hidden layer with 6 units, activation is RELU; third layer is output layer with 1 unit, activation is Sigmoid because the output is binary. Moreover, I add regularization on kernels to compute loss = l2 * reduce_sum(square(x)). For Decision Tree & Random Forest: I divide dataset into 3 sets (train, cross validation and test) to reduce overfitting and underfitting likelihood. XGBoost performs just as well as Random Forest.
- Step 6: Implement model - I test the model with new data input to see how much probability the model gives me for customer's likelihood of churn.

**4. KMeans** 

Case study: Customer segmentation request based on data in Gender, Age, Annual Income (k$), Spending Score (1-100). However, the most important feature here which I want customer group to be defined on is Spending Score. As I calculate the correlation to see the relationship among Spending Score and other information, I see that there's negative correlation between it and Age. Therefore, I use simple KMeans algorithms based on these two values with following steps.
- Step 1: I randomly assign points to cluster centroids where c(i) is the initial centroid
- Step 2: I rearrange centroids  to minimize the distance between x(i) and u(cluster)
- Step 3: I visualized all data points with contour line between each group.
