# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading cars data
carsData = pd.read_csv("car.data", encoding = "latin-1")

# Gathering the information of the data set
carsData.shape
carsData.columns
carsData.describe()
carsData.info()

# Gathering the uniquie present in the columns

for i in carsData.columns:
    print(carsData[i].unique(),"\t",carsData[i].nunique())
    
# Count the columnds dates
    
for i in carsData.columns:
    print(carsData[i].value_counts())
    print()
    

# Seabormn graps for distribution
sns.countplot(carsData['distribution'])



print("printing features graph")
for i in carsData.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("feature '%s'"%i)
    sns.countplot(carsData[i],hue=carsData['distribution'])
    

# Encoding the the data set
"""
buying: buying price – vhigh -> 4, high -> 3, med -> 2, low -> 1
maint: maintenance – vhigh -> 4, high -> 3, med -> 2, low -> 1
doors: number of doors – 2 -> 2, 3 -> 3, 4 -> 4, 5more -> 5
persons: persons capacity – 2 -> 2, 4 -> 4, more -> 5
lug_boot: trunk size – small -> 1, med -> 2, big -> 3
safety: safety rating – small -> 1, med -> 2, big -> 3
"""
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


for i in carsData.columns:
    carsData[i]=le.fit_transform(carsData[i])
    
carsData.head()

# Heat map with corelation 
#Heatmap of the columns on dataset with each other. 
#It shows Pearson's correlation coefficient of column w.r.t other columns.
fig=plt.figure(figsize=(10,6))
sns.heatmap(carsData.corr(),annot=True)

#X is the dataframe containing input data / features

#y is the series which has results which are to be predicted.

X=carsData[carsData.columns[:-1]]
y=carsData['distribution']

X.describe()
X.info()
X.columns
X.dtypes
# Converting to the 64 bit to avoid in64Index issues
X=X.astype(np.int64)
X.dtypes

from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import learning_curve,validation_curve
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


kf = KFold(n_splits=10, shuffle=True)
scoresLR=[]
for train_index, test_index in kf.split(X):
    X_train,X_test,  y_train, y_test = X.loc[train_index],X.loc[test_index], y[train_index], y[test_index]
    logreg=LogisticRegression(solver='newton-cg',multi_class='multinomial')
    logreg.fit(X_train,y_train)
    scoresLR.append(logreg.score(X_test,y_test))
    pred=logreg.predict(X_test)
    confusion_matrix(y_test,pred)
    param_range=[0.0001,0.001,0.1,1]
    curve=validation_curve(logreg,X_train,y_train,cv=5,param_name='C',
    param_range=param_range,n_jobs=-1,)
    try:
        lc=learning_curve(logreg,X_train,y_train,cv=10,n_jobs=-1)
        size=lc[0]
        train_score=[lc[1][i].mean() for i in range (0,5)]
        test_score=[lc[2][i].mean() for i in range (0,5)]
        fig=plt.figure(figsize=(12,8))
        plt.plot(size,train_score)
        plt.plot(size,test_score)
        n=len(param_range)
        train_score=[curve[0][i].mean() for i in range (0,n)]
        test_score=[curve[1][i].mean() for i in range (0,n)]
        ig=plt.figure(figsize=(8,6))
        plt.plot(param_range,train_score)
        plt.plot(param_range,test_score)
        plt.xticks=param_range
    except:
        print("Kflod Spliting issues")
    

print("--scoresLR--")
print(scoresLR)


"""
logreg=LogisticRegression(solver='newton-cg',multi_class='multinomial')

logreg.fit(X_train,y_train)

pred=logreg.predict(X_test)

confusion_matrix(y_test,pred)

logreg.score(X_test,y_test)

from sklearn.model_selection import learning_curve,validation_curve
param_range=[0.0001,0.001,0.1,1]
curve=validation_curve(logreg,X_train,y_train,cv=5,param_name='C',
    param_range=param_range,n_jobs=-1,)


lc=learning_curve(logreg,X_train,y_train,cv=10,n_jobs=-1)
size=lc[0]
train_score=[lc[1][i].mean() for i in range (0,5)]
test_score=[lc[2][i].mean() for i in range (0,5)]
fig=plt.figure(figsize=(12,8))
plt.plot(size,train_score)
plt.plot(size,test_score)


n=len(param_range)
train_score=[curve[0][i].mean() for i in range (0,n)]
test_score=[curve[1][i].mean() for i in range (0,n)]
fig=plt.figure(figsize=(8,6))
plt.plot(param_range,train_score)
plt.plot(param_range,test_score)
plt.xticks=param_range

"""