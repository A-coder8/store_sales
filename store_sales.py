# import a immportant librarys
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# read data and fill a nan
df = pd.read_csv("~/myproject/Model/store_sales/store_sales_prediction_1500.csv")
df.fillna(None)

# set X and Y data
x = df.drop(columns=["id","date","customer_age"])
X = pd.get_dummies(x, drop_first=True)
Y = df["sales"]

# train and test split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=42)

# create a model
model = linear_model.LinearRegression()
model.fit(train_x, train_y)

# test model
yhat = model.predict(test_x)

# show model score with R2Score
print("R2Score =",r2_score(test_y, yhat))
