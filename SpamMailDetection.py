# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:22:43 2019

@author: Sai Kiran
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB,GaussianNB, BernoulliNB
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# Reading the Data Set
data = pd.read_csv("spam.csv", encoding = "latin-1" ,nrows=500)
data = data[['v1', 'v2']]
data_spam = data.rename(columns = {'v1': 'label', 'v2': 'text'})

# Converting to categorical 
data_spam.loc[data_spam["label"] == 'ham',"Status"]=1
data_spam.loc[data_spam["label"] == 'spam',"Status"]=0

data_spam.shape

# Data Details description
data_spam.describe()



# Text Cleaning

def cleanTheText(text):
    # Removing punctations
    # Removing Stopwrods
    # return a list clean text words
    
    punctations = [char for char in text if char not in string.punctuation]
    punctations = ''.join(punctations)
    
    # clean words
    cleanWords = [word for word in punctations.split() if word.lower() not in stopwords.words('english')]
    cleantext  = " ".join([word for word in cleanWords])
    return cleantext


data_spam['text'] = data_spam['text'].apply(cleanTheText)

# Preparing the X and Y Variables
data_spam_x = data_spam['text']
data_spam_y = data_spam['Status']

# Applying the TFIDF

tfIdf = TfidfVectorizer(min_df=1,stop_words='english')




kf = KFold(n_splits=10, random_state=42, shuffle=False)
scoresGaussianNB=[]
scoresMultinomialNB=[]
scoresBernoulliNB=[]
for train_index, test_index in kf.split(data_spam_x):
     X_trainKF,X_testKF,  y_trainKF, y_testKF = data_spam_x[train_index],data_spam_x[test_index], data_spam_y[train_index], data_spam_y[test_index]
     X_trainKFCV = tfIdf.fit_transform(X_trainKF)
     trainArrayKF = X_trainKFCV.toarray()
     tfIdf.inverse_transform(trainArrayKF[0])
     #trainArrayKF.iloc[0]
     y_trainKF = y_trainKF.astype('int')
     X_testKF = tfIdf.transform(X_testKF)
     X_testKF=X_testKF.toarray()
     actualKF = np.array(y_testKF)
     gaussianClasifier = GaussianNB()
     gaussianClasifier.fit(trainArrayKF, y_trainKF)
     scoresGaussianNB.append(gaussianClasifier.score(X_testKF, y_testKF))
     multinomialNBClasifier = MultinomialNB()
     multinomialNBClasifier.fit(trainArrayKF, y_trainKF)
     scoresMultinomialNB.append(multinomialNBClasifier.score(X_testKF, y_testKF))
     bernoulliNBClasifier = BernoulliNB()
     bernoulliNBClasifier.fit(trainArrayKF, y_trainKF)
     scoresBernoulliNB.append(bernoulliNBClasifier.score(X_testKF, y_testKF))
  
print("----------GaussianNB---------")
print(scoresGaussianNB)
print(np.mean(scoresGaussianNB))
print("--------MultinomialNB----------")
print(scoresMultinomialNB)
print(np.mean(scoresMultinomialNB))
print("---------BernoulliNB----------")
print(scoresBernoulliNB)
print(np.mean(scoresBernoulliNB))

data =[scoresGaussianNB,scoresMultinomialNB,scoresBernoulliNB]
df = pd.DataFrame(data) 
df.hist()

plt.show()


df.plot.box()
plt.boxplot(df) 
plt.show() 


"""

X_train, X_test, y_train, y_test = train_test_split(data_spam_x, data_spam_y, test_size = 0.1, random_state = 1)

X_trainCV = tfIdf.fit_transform(X_train)

trainArray = X_trainCV.toarray()

tfIdf.inverse_transform(trainArray[0])

X_train.iloc[0]

y_train = y_train.astype('int')
X_testcv = tfIdf.transform(X_test)
X_testcv=X_testcv.toarray()
actual = np.array(y_test)

gaussianClasifier = GaussianNB()
gaussianClasifier.fit(trainArray, y_train)
predction = gaussianClasifier.predict(X_testcv)

print("------GaussianNB-------------")
print(confusion_matrix(actual, predction))

plt.scatter(actual, predction)
plt.xlabel("True Values")
plt.ylabel("Predictions")


multinomialNBClasifier = MultinomialNB()
multinomialNBClasifier.fit(trainArray, y_train)
predction = multinomialNBClasifier.predict(X_testcv)

print("------MultinomialNB-------------")
print(confusion_matrix(actual, predction))

plt.scatter(actual, predction)
plt.xlabel("True Values")
plt.ylabel("Predictions")



bernoulliNBClasifier = BernoulliNB()
bernoulliNBClasifier.fit(trainArray, y_train)
predction = bernoulliNBClasifier.predict(X_testcv)

print("------BernoulliNB-------------")
print(confusion_matrix(actual, predction))

plt.scatter(actual, predction)
plt.xlabel("True Values")
plt.ylabel("Predictions")

kf = KFold(n_splits=10, random_state=42, shuffle=False)
scores=[]
for train_index, test_index in kf.split(data_spam_x):
     X_trainKF,X_testKF,  y_trainKF, y_testKF = data_spam_x[train_index],data_spam_x[test_index], data_spam_y[train_index], data_spam_y[test_index]
     X_trainKFCV = tfIdf.fit_transform(X_trainKF)
     trainArrayKF = X_trainKFCV.toarray()
     tfIdf.inverse_transform(trainArrayKF[0])
     #trainArrayKF.iloc[0]
     y_trainKF = y_trainKF.astype('int')
     X_testKF = tfIdf.transform(X_testKF)
     X_testKF=X_testKF.toarray()
     actualKF = np.array(y_testKF)
     gaussianClasifier = GaussianNB()
     gaussianClasifier.fit(trainArrayKF, y_trainKF)
     scores.append(gaussianClasifier.score(X_testKF, y_testKF))
     print(np.mean(scores))
     
"""
     
     
