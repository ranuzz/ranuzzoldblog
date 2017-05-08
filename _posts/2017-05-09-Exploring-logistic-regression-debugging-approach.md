---
layout: post
title: Exploring logistic regression debugging approach
---

```python
# source : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
```


```python
# all imports 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
np.seterr(all='raise')

from sklearn.linear_model import LogisticRegression
import math
```


```python
#helper functions

def getColumn(ndarray, col, true_val):
    m = len(ndarray)
    ret_col = np.zeros((m,1))
    for i in range(0,m):
        if ndarray[i,col] == true_val:
            ret_col[i,0] = 1
    return ret_col

# sigmoid function
def sigmoid(num):
    #hack
    if (num > 30):
        return 0.99999999999999778
    if (num < -500):
        return 9.0066236945038063e-88
    return 1/(1+np.exp(-num))

sigmoid_v = np.vectorize(sigmoid)

def pow_num(num, p):
    ans = 1
    for i in range(1,p+1):
        ans = ans*num
    return ans

pow_num_v = np.vectorize(pow_num)

# add one extra column per existing column for each new power
def poly_all(x_features, maxpow):
    features = x_features.shape[1]
    for i in range(0, features):
        for p in range(2, maxpow+1):
            x_features = np.c_[x_features, pow_num_v(x_features[:,i], p)]
    return x_features

# naive cost and gradient calculation function
def costFunctionBasic(theta, x_features, y_result):
    m = len(y_result)
    grad = np.zeros((len(theta), 1))
    H = sigmoid_v(np.dot(x_features,theta))
    J = (-1/m) * np.sum(((y_result)*np.log(H)) + ((1-y_result)*np.log(1- H)))
    for i in range(0,len(grad)):
        grad[i] = (1/m) * np.sum((H - y_result)*x_features[:,i].reshape(m,1))
    return J, grad

def trainPlotBasic(iterations, theta, x_features, y_result):
    jhist = []
    iterx = []
    alpha = 0.001
    for i in range(1, iterations):
        cost_out, grad_out = costFunctionBasic(theta, x_features, y_result)
        jhist.append(cost_out)
        iterx.append(i)
        for i in range(0, len(theta)):
            theta[i] = theta[i] - grad_out[i]
    plt.plot(iterx, jhist, 'rx', linewidth=2)
    plt.show()
    hc, hgrad = costFunctionBasic(theta, x_features, y_result)
    print(hc)
    
def lrCostFunctionCost(theta, x_features, y_result, lambda_val):
    m = len(y_result)
    H = sigmoid_v(np.dot(x_features,theta))
    J = (-1/m) * np.sum(((y_result)*np.log(H) + (((1-y_result))*np.log(1 - H))))
    theta_ex = theta[1:]
    J = J + ((lambda_val/(2*m))*np.dot(theta_ex.T,theta_ex).reshape(1,)[0])
    return J

def lrCostFunctionGrad(theta, x_features, y_result, lambda_val, alpha):
    m = len(y_result)
    grad = np.zeros((len(theta), 1))
    H = sigmoid_v(np.dot(x_features,theta))
    cons = alpha*(lambda_val/m)
    grad[0] = alpha*(1/m) * np.sum(((H - y_result)*(x_features[:,0].reshape(m,1))))
    for i in range(1,len(grad)):
        grad[i] = alpha*(1/m) * np.sum((H - y_result)*(x_features[:,i].reshape(m,1)))
        g_ex =  cons * theta[i]
        grad[i] = grad[i] + g_ex
    return grad.reshape((len(theta),))

def lrTrain(iterations, theta, x_features, y_result, lambda_val, alpha, plot=True):
    jhist = []
    iterx = []
    for i in range(1, iterations):
        cost_out = lrCostFunctionCost(theta, x_features, y_result, lambda_val)
        grad_out = lrCostFunctionGrad(theta, x_features, y_result, lambda_val, alpha)
        jhist.append(cost_out)
        iterx.append(i)
        for i in range(0, len(theta)):
            theta[i] = theta[i] - grad_out[i]
    if (plot):
        plt.plot(iterx, jhist, 'rx', linewidth=2)
        plt.show()
        hc = lrCostFunctionCost(theta, x_features, y_result, lambda_val)
        print(hc)
    return theta
    
def featureNormalize(x_features):
    m = len(x_features)
    features = x_features.shape[1]
    mu = np.apply_along_axis(np.mean, 0, x_features).reshape((features,1))
    norm_features = np.zeros((m, features))
    for i in range(0,features):
        norm_features[:,i] = (x_features[:,i].reshape((m,1)) - (np.ones((m,1))*mu[i])).reshape((m,))
    sigma = np.apply_along_axis(np.std, 0, x_features).reshape((features,1))
    for i in range (0, features):
        norm_features[:,i] = (norm_features[:,i].reshape((m,1)) - (np.ones((m,1))*sigma[i])).reshape((m,))
    return norm_features[:,0:features+1], mu, sigma

def featureNormalizePredefined(x_features, mu, sigma):
    m = len(x_features)
    features = x_features.shape[1]
    norm_features = np.zeros((m, features))
    for i in range(0,features):
        norm_features[:,i] = (x_features[:,i].reshape((m,1)) - (np.ones((m,1))*mu[i])).reshape((m,))
    for i in range (0, features):
        norm_features[:,i] = (norm_features[:,i].reshape((m,1)) - (np.ones((m,1))*sigma[i])).reshape((m,))
    return norm_features[:,0:features+1]


def learningCurve(Xtrain, Ytrain, Xcv, Ycv, lambda_val, iterations, theta, alpha):
    error_train = []
    error_val = []
    max_train = len(Xtrain)
    for i in range(1, max_train+1):
        theta_pred = lrTrain(iterations, theta, Xtrain[0:i,], Ytrain[0:i,], lambda_val, alpha, plot=False)
        cost_train = lrCostFunctionCost(theta_pred, Xtrain[0:i,], Ytrain[0:i,], lambda_val)
        cost_val = lrCostFunctionCost(theta_pred, Xcv, Ycv, lambda_val)
        error_train.append(cost_train)
        error_val.append(cost_val)
    x_axis = np.arange(1, len(error_train)+1)
    plt.plot(x_axis, error_train, 'r-', linewidth=2)
    plt.ylabel('Learning curve for linear regression');
    plt.xlabel('Number of training examples'); 
    plt.plot(x_axis, error_val, 'y-', linewidth=2)
    plt.show()
    return

def getcost_srm(y_pred, y_actual):
    m = len(y_pred)
    cost = 0
    for i in range(0, m):
        cost = cost + math.pow((y_pred[i] - y_actual[i]),2)
    return (math.sqrt(cost) / m)

```


```python
# data input
data_df = pd.read_csv('data.csv', delimiter=',')
data_mat = data_df.values
print(data_mat.shape)
```

    (569, 33)
    


```python
# preview
data_df[5:15] #slective  display
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>843786</td>
      <td>M</td>
      <td>12.45</td>
      <td>15.70</td>
      <td>82.57</td>
      <td>477.1</td>
      <td>0.12780</td>
      <td>0.17000</td>
      <td>0.15780</td>
      <td>0.08089</td>
      <td>...</td>
      <td>23.75</td>
      <td>103.40</td>
      <td>741.6</td>
      <td>0.1791</td>
      <td>0.5249</td>
      <td>0.5355</td>
      <td>0.17410</td>
      <td>0.3985</td>
      <td>0.12440</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>844359</td>
      <td>M</td>
      <td>18.25</td>
      <td>19.98</td>
      <td>119.60</td>
      <td>1040.0</td>
      <td>0.09463</td>
      <td>0.10900</td>
      <td>0.11270</td>
      <td>0.07400</td>
      <td>...</td>
      <td>27.66</td>
      <td>153.20</td>
      <td>1606.0</td>
      <td>0.1442</td>
      <td>0.2576</td>
      <td>0.3784</td>
      <td>0.19320</td>
      <td>0.3063</td>
      <td>0.08368</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>84458202</td>
      <td>M</td>
      <td>13.71</td>
      <td>20.83</td>
      <td>90.20</td>
      <td>577.9</td>
      <td>0.11890</td>
      <td>0.16450</td>
      <td>0.09366</td>
      <td>0.05985</td>
      <td>...</td>
      <td>28.14</td>
      <td>110.60</td>
      <td>897.0</td>
      <td>0.1654</td>
      <td>0.3682</td>
      <td>0.2678</td>
      <td>0.15560</td>
      <td>0.3196</td>
      <td>0.11510</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>844981</td>
      <td>M</td>
      <td>13.00</td>
      <td>21.82</td>
      <td>87.50</td>
      <td>519.8</td>
      <td>0.12730</td>
      <td>0.19320</td>
      <td>0.18590</td>
      <td>0.09353</td>
      <td>...</td>
      <td>30.73</td>
      <td>106.20</td>
      <td>739.3</td>
      <td>0.1703</td>
      <td>0.5401</td>
      <td>0.5390</td>
      <td>0.20600</td>
      <td>0.4378</td>
      <td>0.10720</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>84501001</td>
      <td>M</td>
      <td>12.46</td>
      <td>24.04</td>
      <td>83.97</td>
      <td>475.9</td>
      <td>0.11860</td>
      <td>0.23960</td>
      <td>0.22730</td>
      <td>0.08543</td>
      <td>...</td>
      <td>40.68</td>
      <td>97.65</td>
      <td>711.4</td>
      <td>0.1853</td>
      <td>1.0580</td>
      <td>1.1050</td>
      <td>0.22100</td>
      <td>0.4366</td>
      <td>0.20750</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>845636</td>
      <td>M</td>
      <td>16.02</td>
      <td>23.24</td>
      <td>102.70</td>
      <td>797.8</td>
      <td>0.08206</td>
      <td>0.06669</td>
      <td>0.03299</td>
      <td>0.03323</td>
      <td>...</td>
      <td>33.88</td>
      <td>123.80</td>
      <td>1150.0</td>
      <td>0.1181</td>
      <td>0.1551</td>
      <td>0.1459</td>
      <td>0.09975</td>
      <td>0.2948</td>
      <td>0.08452</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>84610002</td>
      <td>M</td>
      <td>15.78</td>
      <td>17.89</td>
      <td>103.60</td>
      <td>781.0</td>
      <td>0.09710</td>
      <td>0.12920</td>
      <td>0.09954</td>
      <td>0.06606</td>
      <td>...</td>
      <td>27.28</td>
      <td>136.50</td>
      <td>1299.0</td>
      <td>0.1396</td>
      <td>0.5609</td>
      <td>0.3965</td>
      <td>0.18100</td>
      <td>0.3792</td>
      <td>0.10480</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>846226</td>
      <td>M</td>
      <td>19.17</td>
      <td>24.80</td>
      <td>132.40</td>
      <td>1123.0</td>
      <td>0.09740</td>
      <td>0.24580</td>
      <td>0.20650</td>
      <td>0.11180</td>
      <td>...</td>
      <td>29.94</td>
      <td>151.70</td>
      <td>1332.0</td>
      <td>0.1037</td>
      <td>0.3903</td>
      <td>0.3639</td>
      <td>0.17670</td>
      <td>0.3176</td>
      <td>0.10230</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>846381</td>
      <td>M</td>
      <td>15.85</td>
      <td>23.95</td>
      <td>103.70</td>
      <td>782.7</td>
      <td>0.08401</td>
      <td>0.10020</td>
      <td>0.09938</td>
      <td>0.05364</td>
      <td>...</td>
      <td>27.66</td>
      <td>112.00</td>
      <td>876.5</td>
      <td>0.1131</td>
      <td>0.1924</td>
      <td>0.2322</td>
      <td>0.11190</td>
      <td>0.2809</td>
      <td>0.06287</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>84667401</td>
      <td>M</td>
      <td>13.73</td>
      <td>22.61</td>
      <td>93.60</td>
      <td>578.3</td>
      <td>0.11310</td>
      <td>0.22930</td>
      <td>0.21280</td>
      <td>0.08025</td>
      <td>...</td>
      <td>32.01</td>
      <td>108.80</td>
      <td>697.7</td>
      <td>0.1651</td>
      <td>0.7725</td>
      <td>0.6943</td>
      <td>0.22080</td>
      <td>0.3596</td>
      <td>0.14310</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>




```python
# data distribution : 60% train 20% cross validation and 20% test

Y = getColumn(data_mat, 1, 'M').astype(np.float64)

feature_set = data_mat[:,2:-1]
feature_set = feature_set.astype(np.float64)

X = feature_set

Xtrain = X[0:340]
Ytrain = Y[0:340]

Xval = X[340:454]
Yval = Y[340:454]

Xtest = X[454:569]
Ytest = Y[454:569]
```


```python
# simple gradient descent without any normalization/regularization on a linear function 
Xtrain_naive = np.c_[np.ones((len(Xtrain), 1)), Xtrain]
theta = np.zeros((Xtrain_naive.shape[1], 1))
# print(costFunctionBasic(theta, Xtrain, Ytrain))
# initial cost with simple line : 0.69314718055994529
iterations = 100
trainPlotBasic(iterations, theta, Xtrain_naive, Ytrain)
# final cost  : 17.5652032012
```


![png](public/output_6_0.png)


    17.5652032012
    


```python
# lamda : regularization
Xtrain_lambda = np.c_[np.ones((len(Xtrain), 1)), Xtrain]
theta = np.zeros((Xtrain_lambda.shape[1], 1))
#lrCostFunctionCost(theta, Xtrain, Ytrain, 0)
#lrCostFunctionGrad(theta, Xtrain, Ytrain, 0)
iterations = 1000
lambda_val = 10
alpha = 0.03
lrTrain(iterations, theta, Xtrain_lambda, Ytrain, lambda_val, alpha)

# Cost : 2.37355571949
```


![png](public/output_7_0.png)


    160.526793317
    




    array([[ -1.76169066e+00],
           [ -8.05504712e+00],
           [ -1.11720283e+01],
           [ -4.75875033e+01],
           [ -3.23590072e+01],
           [ -7.79018985e-02],
           [  1.65497695e-02],
           [  9.95711315e-02],
           [  4.74403613e-02],
           [ -1.52357530e-01],
           [ -6.02914767e-02],
           [ -1.55241595e-02],
           [ -9.20818163e-01],
           [  3.57864352e-01],
           [  2.20402112e+01],
           [ -5.00831414e-03],
           [  2.51569424e-03],
           [  2.71240880e-03],
           [ -8.52539632e-04],
           [ -1.58521991e-02],
           [ -2.10039034e-03],
           [ -8.21661811e+00],
           [ -1.46369537e+01],
           [ -4.67239100e+01],
           [  3.12756286e+01],
           [ -1.00338233e-01],
           [  9.44196386e-02],
           [  1.95382645e-01],
           [  5.05882047e-02],
           [ -2.05374348e-01],
           [ -5.90371759e-02]])




```python
# feature normalization

Xtrain_normalized, mu, sigma = featureNormalize(Xtrain)
Xtrain_normalized = np.c_[np.ones((len(Xtrain_normalized),1)), Xtrain_normalized]
theta = np.zeros((Xtrain_normalized.shape[1], 1))
iterations = 100
lambda_val = 10
alpha = 0.01
lrTrain(iterations, theta, Xtrain_normalized, Ytrain, lambda_val, alpha)
```


![png](public/output_8_0.png)


    38.6160109598
    




    array([[  1.56672428e-01],
           [  7.59894360e-02],
           [  6.65573826e-02],
           [  6.83615375e-01],
           [ -5.05709804e-01],
           [ -6.70019838e-05],
           [  2.28785246e-03],
           [  3.35545140e-03],
           [  2.75149720e-03],
           [ -1.01283694e-03],
           [ -8.78256956e-04],
           [ -9.11205330e-03],
           [ -7.99281914e-02],
           [ -5.78977078e-02],
           [ -2.09783979e+00],
           [ -5.45545324e-04],
           [ -1.16631471e-03],
           [ -3.23362131e-03],
           [ -1.75830647e-04],
           [ -1.62198897e-03],
           [ -3.81589428e-04],
           [  1.62533141e-01],
           [  2.23532444e-01],
           [  1.32019768e+00],
           [ -3.57482442e-02],
           [  5.51544909e-04],
           [  9.06269864e-03],
           [  1.17242543e-02],
           [  6.73844320e-03],
           [  2.97136552e-04],
           [ -1.51707426e-04]])




```python
# higher order functions : non linear

# helper functions
def selectFeatures(x_features):
    x_features_small = np.c_[x_features[:,0], x_features[:,1],
                    x_features[:,4], x_features[:,5],
                    x_features[:,6], x_features[:,7],
                    x_features[:,8], x_features[:,9]]
    return x_features_small


# feature selection
Xtrainsmall = selectFeatures(Xtrain)
Xtrain_normalized, mu, sigma = featureNormalize(Xtrainsmall)
Xtrain_normalized = poly_all(Xtrain_normalized, 1)
Xtrain_normalized = np.c_[np.ones((len(Xtrain_normalized),1)), Xtrain_normalized]

Xvalsmall = selectFeatures(Xval)
Xval_normalized = featureNormalizePredefined(Xvalsmall, mu, sigma)
Xval_normalized = poly_all(Xval_normalized, 1)
Xval_normalized = np.c_[np.ones((len(Xval_normalized),1)), Xval_normalized]

# tuning parameters
theta = np.zeros((Xtrain_normalized.shape[1], 1))
iterations = 1000
lambda_val = 1
alpha = 0.001

# training algo
#lrTrain(iterations, theta, Xtrain_normalized, Ytrain, lambda_val, alpha)

#learningCurve(Xtrain_normalized, Ytrain, Xval_normalized, Yval, lambda_val, iterations, theta, alpha)

theta_pred = lrTrain(iterations, theta, Xtrain_normalized, Ytrain, lambda_val, alpha)


Xtestsmall = selectFeatures(Xtest)
Xtest_normalized = featureNormalizePredefined(Xtestsmall, mu, sigma)
Xtest_normalized = poly_all(Xtest_normalized, 1)
Xtest_normalized = np.c_[np.ones((len(Xtest_normalized),1)), Xtest_normalized]

print(lrCostFunctionCost(theta_pred, Xtest_normalized, Ytest, lambda_val))

```


![png](public/output_9_0.png)


    0.497460480159
    0.325746064892
    


```python
# square root mean using basic gradient descent

y_pred_lr = np.dot(Xtest_normalized, theta_pred).shape
cost = getcost_srm(y_pred_lr, Ytest)
print(cost)
```

    57.50217387195027
    


```python
# cross check with scikit-learn SVM 

x_svm = Xtrain_normalized
y_svm = Ytrain.reshape((len(Ytrain),))

lr = LogisticRegression(penalty='l2',
                        dual=False,
                        tol=0.000001,
                        C=10.0,
                        fit_intercept=True,
                        intercept_scaling=1,
                        class_weight=None,
                        random_state=1,
                        solver='newton-cg',
                        max_iter=100,
                        multi_class='multinomial',
                        verbose=0,
                        warm_start=False,
                        n_jobs=1)
lr.fit(x_svm, y_svm)
y_pred = lr.predict(Xtest_normalized)

# calculating srm 
cost = getcost_srm(y_pred, Ytest)
print(cost)
```

    0.03253615118933862
    
