#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries required: numpy, scikit-learn
#installing required pip packages in the current Jupyter kernel

import sys
get_ipython().system('{sys.executable} -m pip install numpy scikit-learn')


# In[2]:


#importing modules
import numpy as np                                              #for loading dataset and handling arrays

from sklearn.ensemble import RandomForestClassifier as rfc      #for using random forest classifier model to predict
from sklearn import tree                                        #for using decision tree model to predict
from sklearn.linear_model import LogisticRegression as lr       #for using logistic regression model to predict
from sklearn import svm                                         #for using support vector machines model to predict
from sklearn.neighbors import KNeighborsClassifier              #for using k-nearest neighbour model to predict
from sklearn.model_selection import train_test_split            #for dividing dataset into training and testing 
from sklearn.metrics import accuracy_score                      #for calculating the accuracy score of models
from sklearn.metrics import precision_score                     #for calculating the precision score of models
from sklearn.metrics import recall_score                        #for calculating the recall score of models
from sklearn.metrics import f1_score                            #for calculating the f1 score of models

import time                                                     #for calculating time of training and testing of model


# In[3]:


def LogisticRegression():
    
    #loading dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")
    
    #seperate features and labels, 1-30 are features and 31 is result (label)
    x = data[: , :-1]
    y = data[: , -1]
    
    #Seperating training features, testing features, training labels & testing labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    #here 20% data is for testing
    #x variables contain features and y contains results
    
    print("Training a logistic regression model on given dataset")
    start = time.time()                                         #store the start time for training and testing of model
    classifier = lr()                                           #using logistic regression model
    print("Logistic regression classifier created.")
    print("Beginning model training.")
    classifier.fit(x_train, y_train)                            #train the model
    print("Model training completed.")
    predictions = classifier.predict(x_test)                    #do predictions on the model for testing data
    print("Predictions on testing data computed.")
    end = time.time ()                                          #store the end time for training and testing of model
    accuracy = 100.0 * accuracy_score(y_test, predictions)
    print("The accuracy of your logistic regression model on testing data is: " + str(accuracy) + " %")
    f1score = f1_score (y_test, predictions)
    print ("The f1-score of your logistic regression model on testing data is: " + str (f1score))
    precision = precision_score (y_test, predictions)
    print ("The precision of your logistic regression model on testing data is: " + str (precision))
    recall = recall_score (y_test, predictions)
    print ("The recall of your logistic regression model on testing data is: " + str (recall))
    runtime = end - start
    print ("Total time taken for training and testing by logistic regression model is: " + str (runtime) + " s")


# In[4]:


def DecisionTree():
    
    #loading dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")
    
    #seperate features and labels, 1-30 are features and 31 is result (label)
    x = data[: , :-1]
    y = data[: , -1]
    
    #Seperating training features, testing features, training labels & testing labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    #here 20% data is for testing
    #x variables contain features and y contains results
    
    print("Training a decision tree model on given dataset")
    start = time.time ()
    classifier = tree.DecisionTreeClassifier()
    print("Decision tree classifier created.")
    print("Beginning model training.")
    classifier.fit(x_train, y_train)
    print("Model training completed.")
    predictions = classifier.predict(x_test)
    print("Predictions on testing data computed.")
    end = time.time ()
    accuracy = 100.0 * accuracy_score(y_test, predictions)
    print("The accuracy of your decision tree model on testing data is : " + str(accuracy) + " %")
    f1score = f1_score (y_test, predictions)
    print ("The f1-score of your decision tree model on testing data is: " + str (f1score))
    precision = precision_score(y_test, predictions)
    print("The precision of your decision tree model on testing data is: " + str(precision))
    recall = recall_score(y_test, predictions)
    print ("The recall of your decision tree model on testing data is: " + str (recall))
    runtime = end - start
    print ("Total time taken for training and testing by decision tree model is: " + str(runtime) + " s")


# In[5]:


def RandomForestClassifer():
    
    #loading dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")
    
    #seperate features and labels, 1-30 are features and 31 is label
    x = data[: , :-1]
    y = data[: , -1]
    
    #Seperating training features, testing features, training labels & testing labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    #here 20% data is for testing
    #x variables contain features and y contains results
    
    print("Training a random forest model on given dataset")
    start = time.time()                                         #store the start time for training and testing of model
    classifier = rfc()                                          #using random forest classifier model
    print("Random Forest classifier created.")
    print("Beginning model training.")
    classifier.fit(x_train, y_train)                            #train the model
    print("Model training completed.")
    predictions = classifier.predict(x_test)                    #do predictions on the model for testing data
    print("Predictions on testing data computed.")
    end = time.time ()                                          #store the end time for training and testing of model
    accuracy = 100.0 * accuracy_score(y_test, predictions)      #calculate accuracy of the model and store it in 'score' variable
    print("The accuracy of your random forest model on testing data is: " + str(accuracy) + " %")
    f1score = f1_score (y_test, predictions)            #calculate f1 score of the model and store it in 'f1score' variable
    print ("The f1-score of your random forest model on testing data is: " + str (f1score))
    precision = precision_score (y_test, predictions)   #calculate precision score of the model and store it in 'precision' variable
    print ("The precision of your random forest model on testing data is: " + str (precision))
    recall = recall_score (y_test, predictions)         #calculate recall score of the model and store it in 'recall' variable
    print ("The recall of your random forest model on testing data is: " + str (recall))
    runtime = end - start                                       #calculate and store total time taken for training and testing of model
    print ("Total time taken for training and testing by random forest model is: " + str (runtime) + " s")


# In[6]:


def SupportVectorMachines():
    
    #loading dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")
    
    #seperate features and labels, 1-30 are features and 31 is result
    x = data[: , :-1]
    y = data[: , -1]
    
    #Seperating training features, testing features, training labels & testing labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    #here 20% data is for testing
    #x variables contain features and y contains results
    
    print("Training a support vector machine model on given dataset")
    start = time.time ()
    classifier = svm.SVC()
    print("Support Vector Machines created.")
    print("Beginning model training.")
    classifier.fit(x_train, y_train)
    print("Model training completed.")
    predictions = classifier.predict(x_test)
    print("Predictions on testing data computed.")
    end = time.time ()
    accuracy = 100.0 * accuracy_score(y_test, predictions)
    print("The accuracy of your support vector machines model on testing data is: " + str(accuracy) + " %")
    f1score = f1_score (y_test, predictions)
    print ("The f1-score of your support vector machines model on testing data is: " + str (f1score))
    precision = precision_score (y_test, predictions)
    print ("The precision of your support vector machines model on testing data is: " + str (precision))
    recall = recall_score (y_test, predictions)
    print ("The recall of your support vector machines model on testing data is: " + str (recall))
    runtime = end - start
    print ("Total time taken for training and testing by support vector machines model is: " + str (runtime) + " s")


# In[7]:


def KNearestNeighbour():
    
    #loading dataset
    data = np.loadtxt("dataset.csv", delimiter = ",")
    
    #seperate features and labels, 1-30 are features and 31 is result
    x = data[: , :-1]
    y = data[: , -1]
    
    #Seperating training features, testing features, training labels & testing labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)    #here 20% data is for testing
    #x variables contain features and y contains results
    
    print("Training a k nearest neighbours model on given dataset")
    start = time.time ()
    classifier = KNeighborsClassifier()
    print("K-Nearest Neighbours created.")
    print("Beginning model training.")
    classifier.fit(x_train, y_train)
    print("Model training completed.")
    predictions = classifier.predict(x_test)
    print("Predictions on testing data computed.")
    end = time.time ()
    accuracy = 100.0 * accuracy_score(y_test, predictions)
    print("The accuracy of your k-nearest neighbours model on testing data is: " + str(accuracy) + " %")
    f1score = f1_score (y_test, predictions)
    print ("The f1-score of your k-nearest neighbours model on testing data is: " + str (f1score))
    precision = precision_score (y_test, predictions)
    print ("The precision of your k-nearest neighbours model on testing data is: " + str (precision))
    recall = recall_score (y_test, predictions)
    print ("The recall of your k-nearest neighbours model on testing data is: " + str (recall))
    runtime = end - start
    print ("Total time taken for training and testing by k-nearest neighbours model is: " + str (runtime) + " s")


# In[8]:


#calling for using logistic regression model
LogisticRegression()


# In[9]:


#calling for using decision tree model
DecisionTree()


# In[10]:


#calling for using random forest model
RandomForestClassifer()


# In[11]:


#calling for using support vctor machines model
SupportVectorMachines()


# In[12]:


#calling for using k-nearest neighbours model
KNearestNeighbour()

