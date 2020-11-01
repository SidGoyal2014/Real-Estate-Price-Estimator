# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 00:25:54 2020

@author: DELL
"""

# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
# from sklearn.preprocessing import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# import pickle

path = os.getcwd()

print(path)

csv_path = path+"\Data"

print(csv_path)

csv_path = csv_path+"\\final.csv"

print(csv_path)

# Loading the dataframe
df = pd.read_csv(csv_path)

x1 = df.drop("SALES_PRICE",axis=1)

# Getting the dummies
df = pd.get_dummies(df,dummy_na=True)

# Seperate the dependent & target variables
x = df.drop("SALES_PRICE",axis=1)
y = df["SALES_PRICE"]

print()
print()
print(x)
print()
print()
print(x.columns)
print()
print()

x.drop("MZZONE_nan",axis=1,inplace=True)
x.drop("STREET_nan",axis=1,inplace=True)
x.drop("UTILITY_AVAIL_nan",axis=1,inplace=True)
x.drop("BUILDTYPE_nan",axis=1,inplace=True)
x.drop("PARK_FACIL_nan",axis=1,inplace=True)
x.drop("SALE_COND_nan",axis=1,inplace=True)
x.drop("AREA_nan",axis=1,inplace=True)

# Linear Regression Object
linreg = LinearRegression()

# Polynomial Features
poly = PolynomialFeatures(degree = 2)

# Fit_Transform
X = poly.fit_transform(x)

# Linear regression fittiing the model
linreg.fit(X,y)

#linreg.fit(x,y)
print("DEBUGGING : ",x.columns)
#save
import joblib

joblib.dump(linreg,'model.pkl')
print("Model Dumped!")

# Saving the model columns
linreg = joblib.load('model.pkl')

# Saving the data columns for training
model_columns = list(x.columns)

print("###########################################")
print("###########################################")
print(model_columns)

"""
model_columns.remove("MZZONE_nan")
model_columns.remove("STREET_nan")
model_columns.remove("UTILITY_AVAIL_nan")
model_columns.remove("BUILDTYPE_nan")
model_columns.remove("PARK_FACIL_nan")
model_columns.remove("SALE_COND_nan")
model_columns.remove("AREA_nan")
"""

print(model_columns)

joblib.dump(model_columns,'model_columns.pkl')
print("Model Column Dumped")
