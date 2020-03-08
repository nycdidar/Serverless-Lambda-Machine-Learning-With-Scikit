import json
import re
import os

import pickle
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

saved_model = 'data/data_model.model'
data = pd.read_csv('data/iris.csv', sep = ',')

# Preprocess Data
def preprocess_data(column):
    label_identify = LabelEncoder()
    data[column] = label_identify.fit_transform(data[column])
    data[column].value_counts()
    X = data.drop(column, axis =1)
    y = data[column]
    
    # Train Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    #Apply standard scaling to get optimized result
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test, sc


def train_data(column):
    X_train, X_test, y_train, y_test, sc = preprocess_data(column)
    
    # Neural Network
    model = MLPClassifier(hidden_layer_sizes = (11,11,11), max_iter = 1000)
    model.fit(X_train, y_train)
        
    # Save model for future use
    pickle.dump(model, open(saved_model, 'wb'))
    pred_mlpc = model.predict(X_test)
    #print(pred_mlpc)
    print("Done")

train_data('species')