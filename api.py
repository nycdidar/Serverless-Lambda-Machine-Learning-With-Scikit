import json
import re
import os

import urllib

#https://www.dataquest.io/blog/sci-kit-learn-tutorial/

## YOU MAY HAVE TO RUN IT FEW TIMES LOCALLY TO EXECUTE
# this helps us use zipped dependencies on Lambdas,
# to work around numpy/scipy size requirements
# It must run before we import any non standard library requirements on AWS
try:
  print('attempting to unzip_requirements...')
  import subprocess
  print(subprocess.check_output(['df']))
  import unzip_requirements  # noqa
  print('succesfully imported unzip_requirements to prepare the zipped requirements')
except ImportError:
  print('failed to import unzip_requirements - if you are running locally this is not a problem')

### All Non Standard Lamba Libraries Must Be Here.


import pickle
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

saved_model = 'data/data_model.model'
data = pd.read_csv('data/iris.csv', sep = ',')

def GetPostData(body):
    postdata = {}
    for items in body.split('&'):
        vals = items.split('=')
        postdata[vals[0]] = urllib.parse.unquote(vals[1])
    return postdata

# Preprocess Data
def preprocess_data(column, input_data):
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

def classify(column, input_data):
    #X_train = loadtxt('data/train_data.csv', delimiter=',')
    #Apply standard scaling to get optimized result
    
    input_data = [input_data.split(',')]
    X_train, X_test, y_train, y_test, sc = preprocess_data(column, input_data)
        
    # load the model from disk
    model = pickle.load(open(saved_model, 'rb'))
    input_data = sc.transform(input_data)
    result = int(model.predict(input_data))
    
    # Label output data
    switcher = { 
        0: "setosa", 
        1: "versicolor", 
        2: "virginica", 
    }
    return switcher.get(result, "nothing")


def handler(event, context):
  body_data = GetPostData(event['body'])
  resultObject = {
    "flower_type": classify('species', body_data['input_data']),
    "body": body_data['input_data']
  }

  result = json.dumps(resultObject)
  return {"statusCode": 200, "body": result}

