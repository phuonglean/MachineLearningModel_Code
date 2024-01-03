# MachineLearningModel_Code
Standard code for Machine Learning models and will be updated every new model I have learned on Machine Learning Specialization on Coursera.

Last Updated Date: 30.12.2023

I have completed Supervised Learning and Advanced Learning algorithms courses which cover:
#1. Linear Regression (with Gradient Descent, Cost function computation, Feature Scaling, SGDRegression)
#2. Logistic Regression (Batch Gradient Descent, Cost function computation)
#3. Neural Network:
Case study brief: Customer profiles dataset at one bank comprises values of Credit Score, Geography, Gender, Age, Tenure, Balancing, Number of products, Credit Card possession, Active Member, Estimated Salary, Excited. Based on these columns, a bank wants to predict whether the customers will churn.
Approach: To start with simple method, I use ANN model to predict the probability of churn with 6 steps below:
- Step 1: Import libraries data - I use sklearn libraries and tensorflow package whereas the dataset is on Kaggle named Bank Customer Churn Dataset
- Step 2: Explore data - I use visualization to demonstrate distribution, skewness and correlations of variables
- Step 3: Preprocess data - I use one-hot encoding to treat categorical variables. Then splitting dataset into train and test set
- Step 4: Feature scaling - Because each features differ from each other scales, I use standard scaler which subtract value on each row's feature by mean and then divide it by standard deviation as for 2 reasons: first is distributions of all features are not normally distributed; second is each feature has different value range which increase time to reach optimal global minimum of loss function
- Step 5: Build model - I developed simple model with 3 layers: first layer is input layer with 13 units, activation function is RELU; second layer is hidden layer with 6 units, activation is RELU; third layer is output layer with 1 unit, activation is Sigmoid because the output is binary. Moreover, I add regularization on kernels to compute loss = l2 * reduce_sum(square(x))
- Step 6: Implement model - I tested the model with new data input to see how much probability the model gives me for customer's likelihood of churn.
