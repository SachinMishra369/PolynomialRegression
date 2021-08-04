#importing the libraries 
import pandas as pd
import numpy as np
#READING THE DATASET
dataset=pd.read_csv('salary.csv')
print(dataset)

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,-1].values
print(Y)
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,Y)
#POLNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)

x_poly=poly.fit_transform(X)
regressor_2=LinearRegression()
regressor_2.fit(x_poly,Y)

#VISUALZING LINEAR REGRESSION
import matplotlib.pyplot  as plt
# plt.scatter(X,Y,color='blue')
# plt.plot(X,regressor.predict(X),color='red')
# plt.xlabel('Em Level')
# plt.ylabel('Salary')
# plt.show()


#VISUALIZING POLYNOMIAL REGRESSION'

plt.scatter(X,Y,color='blue')
plt.plot(X,regressor_2.predict(x_poly),color='red')
plt.xlabel('Em Level')
plt.ylabel('Salary')
plt.show()
