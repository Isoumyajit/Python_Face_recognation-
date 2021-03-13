import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

# /////..........................
# load Boston datasets from Sklearn Dataset class
from sklearn import datasets

BostonData = datasets.load_boston()
X_FeatureData = pd.DataFrame(BostonData.data , columns = BostonData.feature_names)
# print(X_FeatureData)
Y_ActualOutCome = pd.DataFrame(BostonData.target)
# print(Y_ActualOutCome)

X_train , X_test , Y_train , Y_Test = train_test_split(X_FeatureData , Y_ActualOutCome , test_size=0.35 , random_state=35)
train = linear_model.LinearRegression()
train.fit(X_train , Y_train)

Y_predict = train.predict(X_test)
plt.scatter(Y_Test , Y_predict)
plt.xlabel('Features -- >')
plt.ylabel('Price -- >')
plt.show()
print(Y_Test)

print(Y_predict)
print(mean_squared_error(Y_Test , Y_predict))


