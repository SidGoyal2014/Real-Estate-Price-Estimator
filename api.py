# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 00:59:58 2020

@author: DELL
"""

"""
from flask import Flask,request,jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import requests

# API Definition
app = Flask(__name__)


@app.route('/predict',methods=['POST'])
def predict():
    if(linreg):
        try:
            print("REQUESTED JSON : ",request.json)
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns,fill_value=0)
            
            # print("Query : ",query)
            
            prediction = list(linreg.predict(query))
            
            # print("prediction : ",prediction)
            
            return jsonify({'Prediction : ',str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print("No model is there")
        
if __name__ == '__main__':
    linreg = joblib.load("model.pkl")
    print("Model Loaded")
    model_columns = joblib.load("model_columns.pkl")
    print("Model Columns Loaded")
    app.run(debug=True)
    
"""

# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import PolynomialFeatures

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            print("REQUESTED JSON : ",request.json)
            # print("REQUESTED JSON 1 : ",requests.json)
            json_ = request.json
            
            # print(json_)
            print()
            print()
            
            query = pd.get_dummies(pd.DataFrame(json_))
            
            query = query.reindex(columns=model_columns, fill_value=0)
            
            print()
            print()
            print(query)
            

            poly = PolynomialFeatures(degree = 2)
            
            query = poly.fit_transform(query)
            
            print("###################################")
            print("query : ",query)
            print("Length of query : ",len(query))
            # print("columns : ",query.columns)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:
            print("Netaji")
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(debug = True)