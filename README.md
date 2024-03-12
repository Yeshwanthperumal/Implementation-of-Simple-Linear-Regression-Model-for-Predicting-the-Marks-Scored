# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn
2. Calculate the values of the training data set 
3. Calculate the values of the test data set
4. Plot the graph for both the data sets and calculate MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YESHWANTH P
RegisterNumber:  212222230178
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

##  splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred

## graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
<h3>df.head</h3>

![Screenshot 2024-03-12 195307](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/dba126a3-e9d5-41a6-b784-b9cbc69780db)
<h3>df.tail</h3>

![Screenshot 2024-03-12 195426](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/4ae34996-761a-4a3e-82da-36f67fcefd7b)
<h3>ARRAY VALUE OF X</h3>

![Screenshot 2024-03-12 195956](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/db5ccaae-e694-45f7-929f-b8443e43092c)
<h3>ARRAY VALUE OF Y</h3>

![Screenshot 2024-03-12 200505](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/1c9cf2b9-cbd7-4365-82e8-8acf5a94873f)
<h3>VALUES OF Y PREDICTION</h3>

![Screenshot 2024-03-12 200622](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/0a683f85-87f4-470b-912b-ffec364a46ad)
<h3>ARRAY VALUES OF Y TEST</h3>

![Screenshot 2024-03-12 222548](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/e19aee2e-d2b7-449c-a96b-8af9acd020ac)
<h3>TRAINING SET GRAPH</h3>

![image](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/cc162b75-9fd7-445a-9817-c1c74fe438f2)
<h3>TEST SET GRAPH</h3>

![image](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/71a37551-97df-4bd4-b38d-099ee0b5de6a)
<h3>VALUES OF MSE, MAE AND RMSE</h3>

![image](https://github.com/Yeshwanthperumal/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119476088/5532f38e-4ec4-480e-bd1b-e917e2defdfb)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
