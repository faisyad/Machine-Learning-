import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("D:\Machine Learning\Project+Exercise Single Linear Regression\Indonesiacapita.csv")
df

df.columns.values

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Years')
plt.ylabel('income per capita')
plt.scatter(df.years, df.percapitaincome, color = 'red', marker = '+')

reg = linear_model.LinearRegression()
reg.fit(df[['years']].values, df.percapitaincome)

reg.get_params()

reg.predict([[2022]])

reg.coef_

reg.intercept_

72693.90909091*2022+-142759713.1060606

d=pd.read_csv("D:\Machine Learning\Project+Exercise Single Linear Regression\Indonesiacapita.csv")
d

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Years')
plt.ylabel('Income per Capita')
plt.scatter(d.years, d.percapitaincome, color='red', marker ='.')
plt.plot(d.years, reg.predict(d[['years']]), color='red')
