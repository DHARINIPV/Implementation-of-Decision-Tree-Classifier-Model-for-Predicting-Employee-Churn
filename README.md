# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dharini PV
RegisterNumber:  212222240024
```
```python
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:

## Initial dataset:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/929f47cc-2a5c-4bfb-9bdf-7b24b0755f71)

## Data Info:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/0527e4f7-c2b3-4d10-94c5-86e4d122b601)

## Optimization of null values:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/b98bd12d-e6a6-4170-a9ac-2fe768a39ed8)

## Assignment of x and y values:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/1ebf6e1c-5bd9-4b7e-891a-fddb0e80ad65)

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/2ce0764f-3526-4d5b-8d3d-2c74bf65fafb)

## Converting string literals to numerical values using label encoder:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/3cd00b18-a23c-4c15-a613-0ba8ea0438e4)

## Accuracy:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/fadda795-1243-4cc2-a0cc-4a7441ab4943)

## Prediction:

![image](https://github.com/DHARINIPV/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119400845/3fff2cb3-f34e-4fbb-8e71-9898a6fb6c93)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
