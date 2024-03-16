import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/dbala/Downloads/DataData123.csv", index_col=False)
print(df.head())

y = df["Electricity"]
x = df.drop("Electricity", axis = 1)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train,y_train)
y_prediction = LR.predict(x_test)
#y_prediction
print("Coefficient of lrm", LR.coef_)
print("Intercept of lrm", LR.intercept_)
print("R^2 value", metrics.r2_score(y_test, y_prediction))

xValues = df.iloc[64].to_numpy()
yValues = LR.intercept_


for i in range (0,len(LR.coef_)):
    yValues = yValues + LR.coef_[i]*xValues[i]




