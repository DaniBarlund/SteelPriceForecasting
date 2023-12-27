#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, optimize
import datetime

import warnings
warnings.filterwarnings(action='once')


# In[40]:


# Read data from the file to a dataframe
df = pd.read_csv("Steel Futures Historical Data.csv")
print(df)


# In[41]:


# Find variables which have nans as values
nans = df.columns[df.isna().any()].tolist()
print(nans)


# In[42]:


# Nan in one variable which can be counted easily -> Drop column 'Vol.'
df = df.drop(columns=['Vol.'])
print(df)


# In[43]:


# Make numbers from string to int
for date in df.index:
    df['Price'][date] = df['Price'][date].replace(',','')
    df['Open'][date] = df['Open'][date].replace(',','')
    df['High'][date] = df['High'][date].replace(',','')
    df['Low'][date] = df['Low'][date].replace(',','')


# In[44]:


# Change varibles price, open, high and low from string to integer
df['Price'] = df['Price'].astype(float)


# In[45]:


# Plot the price of steel for date
plt.plot(df.index,df['Price'])
plt.gca().invert_xaxis()
plt.xlabel("index")
plt.ylabel("Price")
plt.show()


# In[46]:


# Check the change from the highest price to now
change = df['Price'].iloc[0]/max(df['Price'])
print(f"Current price to max is {change*100:.2f}%")


# In[47]:


# Try adding a line to data
plt.plot(df.index,df['Price'],label="Price")
y = df['Price']
x = df.index
#Plot lines from different degree polynomials
a, b = np.polyfit(x,y,1)
plt.plot(x,a*x+b,label="Degree of 1")

a, b, c = np.polyfit(x,y,2)
plt.plot(x,a*x**2+b*x+c,label="Degree of 2")

a, b, c, d = np.polyfit(x,y,3)
plt.plot(x,a*x**3+b*x**2+c*x+d,label="Degree of 3")

a, b, c, d, e = np.polyfit(x,y,4)
plt.plot(x,a*x**4+b*x**3+c*x**2+d*x+e,label="Degree of 4")

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x, slope * np.array(x) + intercept, label='Linear regression')

plt.legend()
plt.gca().invert_xaxis()
plt.xlabel("index")
plt.ylabel("Price")
plt.show()


# In[48]:


# Polynomial degree of 3 is best
a, b, c, d = np.polyfit(x,y,3)
print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
print(f"d={d}")


# In[49]:


#Format data to have data as index
dfDate = df
dfDate.index = pd.to_datetime(df['Date'],format='%m/%d/%Y')
dfDate = dfDate.drop(columns=['Date'])
print(dfDate)


# In[50]:


plt.plot(dfDate.index, dfDate['Price'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[51]:


# Values are missing on the straigth line. Change data to start from 2021
dfDate = dfDate[df.index > pd.to_datetime("2021-01-01", format="%Y-%m-%d")]
print(dfDate)


# In[52]:


plt.plot(dfDate.index, dfDate['Price'])
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[53]:


#TRy to fix the problem of having wrong data in prediction
df = df.iloc[::-1]
dfDate = df
dfDate.index = pd.to_datetime(df['Date'],format='%m/%d/%Y')
dfDate = dfDate[df.index > pd.to_datetime("2021-01-01", format="%Y-%m-%d")]


# In[54]:


# Divide into train and test sets
train = dfDate[dfDate.index < pd.to_datetime("2023-10-01", format="%Y-%m-%d")]
test = dfDate[dfDate.index >= pd.to_datetime("2023-10-01", format="%Y-%m-%d")]
plt.plot(train.index,train['Price'], label="train", color="black")
plt.plot(test.index,test['Price'], label="test", color="red")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[55]:


# Forecasting with ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

y = train['Price']
modelARMA = SARIMAX(y, order = (1,0,1))
modelARMA = modelARMA.fit()

yPred = modelARMA.get_forecast(len(test.index))
yPredDf = yPred.conf_int(alpha = 0.05)
yPredDf["Predictions"] = modelARMA.predict(start = yPredDf.index[0], end = yPredDf.index[-1])
yPredDf.index = test.index
predictions = yPredDf["Predictions"]


# In[56]:


plt.plot(train.index,train['Price'], label="train", color="black")

plt.plot(test.index,test['Price'], label="test", color="red")

plt.plot(predictions, color = "orange", label = "predictions")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[57]:


# Count RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test["Price"].values,yPredDf["Predictions"]))
print(f"RMSE: {rmse:.2f}")


# In[58]:


# Find best prediction values
y = train['Price']
best = 1000
values = (0,0,0)
for a in range(5):
    for b in range(5):
        for c in range(5):
            modelARMA = SARIMAX(y, order = (a,b,c))
            modelARMA = modelARMA.fit()
            
            yPred = modelARMA.get_forecast(len(test.index))
            yPredDf = yPred.conf_int(alpha = 0.05)
            yPredDf["Predictions"] = modelARMA.predict(start = yPredDf.index[0], end = yPredDf.index[-1])
            rmse = np.sqrt(mean_squared_error(test["Price"].values,yPredDf["Predictions"]))
            if rmse<best:
                best = rmse
                values = (a,b,c)

print(f"RMSE: {rmse:.2f}")
print(values)


# In[27]:


modelARMA = SARIMAX(y, order = (1,3,3))
modelARMA = modelARMA.fit()

yPred = modelARMA.get_forecast(len(test.index))
yPredDf = yPred.conf_int(alpha = 0.05)
yPredDf["Predictions"] = modelARMA.predict(start = yPredDf.index[0], end = yPredDf.index[-1])
yPredDf.index = test.index
predictions = yPredDf["Predictions"]

plt.plot(train.index,train['Price'], label="train", color="black")

plt.plot(test.index,test['Price'], label="test", color="red")

plt.plot(predictions, color = "orange", label = "predictions")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
plt.show()

rmse = np.sqrt(mean_squared_error(test["Price"].values,yPredDf["Predictions"]))
print(f"RMSE: {rmse:.2f}")


# In[ ]:




