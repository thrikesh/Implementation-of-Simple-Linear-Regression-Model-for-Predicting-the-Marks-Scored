# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
    1.Import the standard Libraries
    2.Set variables for assigning dataset values
    3.Import linear regression from sklearn.
    4.Compare the graphs and hence we obtained the linear regression for the given data  
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THRIKESWAR P
RegisterNumber:  212222230162
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Thrikeswar.P 
RegisterNumber:212222230162
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

#spitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#display predicted values
Y_pred

#display actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="Red")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
*/
```

## Output:
df head()

![df head()](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/bfbab2d5-e452-4d1a-a315-5fa730b09ee6)


df tail()

![df tail()](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/49195657-c6d5-4520-aa94-6b1fe4bd1aae)

Array value of X

![Array value of X](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/0d6092ad-c490-4b80-9796-3a56f7eb50a6)

Array value of y

![Array value of y](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/eb82ee76-4486-43c2-b169-794befbd25db)

values of Y prediction

![values of Y prediction](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/214caf1a-8793-4587-9a00-6b12629b74bc)

Array value of Y test

![Array value of Y test](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/22c675dd-9f58-4b80-99a8-8bbd7ad958ec)

Training set Graph

![Training set Graph](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/2bcca399-7c09-464b-9376-e98c91e741f7)

Test set Graph

![Test set Graph](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/99117449-d4ff-4d58-9e60-06d568771c01)

Values of MSE,MAE and RMSE

![Values of MSE,MAE and RMSE](https://github.com/Naveensrinivasan07/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475891/88d90cc3-44ce-4042-bf49-134d9615ae00)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
