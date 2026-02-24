# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.
## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: POOJA U
RegisterNumber: 212225230209
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('CarPrice_Assignment.csv')

#Select features and target
X = df[['enginesize','horsepower','citympg','highwaympg']]
Y = df['price']

#Split target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#Feature scaling becz it will be easier when evtg is in same format
scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fit the X and Y in a straight line ie we are training the model
model = LinearRegression()
model.fit(X_train_scaled,Y_train)

#Predict the outcome by giving new set of data
Y_pred = model.predict(X_test_scaled)

#intersept ir beta knot values and coefficiets are being displayed
#model coefficient and metrics
print("Name: U POOJA ")
print("Reg. No: 25011745")
print("MODEL COEFFICIENTS: ")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:} : {coef:}")
print(f"{'Intercept':} : {model.intercept_:}")

#Performance metrics
print("MODEL PERFORMANCE:")
mse = mean_squared_error(Y_test,Y_pred)
print(f"MSE : {mse}")
print(f"MAE : {mean_absolute_error(Y_test,Y_pred)}")
print(f"RMSE : {np.sqrt(mse)}")
print(f"R-Squared : {r2_score(Y_test,Y_pred)}")

# 1. Linearity Check
#to check if evtg is in same format
plt.figure(figsize=(10,5))
plt.scatter(Y_test,Y_pred, alpha =0.6)
plt.plot([Y.min(),Y.max()],[Y.min(),Y.max()], 'r--')
plt.title("Lineraity Check : Actual VS Predicted Prices")
plt.xlabel("Acatual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals= Y_test - Y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\n Durbin-Watson Statistic: {dw_test:.2f}",
     "\n (values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=Y_pred, y=residuals,lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals VS Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

#4. Normality of residuals
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde= True, ax=ax1)
ax1.set_title("Resitduals Distribution")
sm.qqplot(residuals, line='45',fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
```

## Output:

<img width="853" height="373" alt="Screenshot 2026-02-07 141235" src="https://github.com/user-attachments/assets/adc03754-dc8f-4560-9651-9ad298cf3ff2" />
<img width="870" height="275" alt="Screenshot 2026-02-07 141247" src="https://github.com/user-attachments/assets/9ada1e91-e019-48c3-aac2-c2bf1eb177fa" />

<img width="1360" height="777" alt="Screenshot 2026-02-07 141305" src="https://github.com/user-attachments/assets/fc232257-8355-4f20-93d5-6e1b40afa4b5" />
<img width="1319" height="753" alt="Screenshot 2026-02-07 141354" src="https://github.com/user-attachments/assets/56944512-907d-4abe-bcfa-1a3cbd694f3b" />

<img width="1356" height="752" alt="Screenshot 2026-02-07 141432" src="https://github.com/user-attachments/assets/34ec7dc7-593f-4804-8534-dd1d63213360" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
