# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values
3. Import linear regression from sklearn.
4. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
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
```

## Output:
df.head()
<br>
![ml ex2(a)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/3ac0875e-41dd-4357-bbb3-08dd596cf525)
<br>
df.tail()
![ml ex2(b)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/665f1e40-7ae3-4000-8daa-c61992b5d538)
Array value of X
![ml ex2(c)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/6eedca23-f1d6-47c7-a0fb-a12765c0b94f)
Array value of Y
![ml ex2(d)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/79fac52f-b0d1-4858-b37f-a623d6158448)
values of Y prediction
![ml ex2(e)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/f54bdec7-1f30-4736-aa0b-5784ca486316)
Array value of Y test
![ml ex2(f)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/2b9a48ae-5afb-4232-9fd4-e23964426dfe)
Training set Graph
![ml ex2(g)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/c26f5d3c-459a-44d8-8bb8-69a2da30f73d)
Test set Graph
![ml ex(i)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/8472f819-cdfd-49c1-8ae4-fad62015e919)
Values of MSE,MAE and RMSE
![ml ex(h)](https://github.com/thrikesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119576222/0ce028e7-7bc9-47d8-a3b0-7037457e175f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
