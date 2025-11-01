# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 01-10-25

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# --- Step 1: Load dataset ---
data = pd.read_csv('usedcarssold.csv')

# --- Step 2: Preprocess ---
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data.dropna(subset=['Sold_Cars'])  # Remove rows with missing values

# --- Step 3: Plot time series ---
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Sold_Cars'], label='Sold_Cars', color='blue')
plt.xlabel('Date')
plt.ylabel('Number of Cars Sold')
plt.title('Used Cars Sold Over Time')
plt.legend()
plt.show()

# --- Step 4: Check stationarity ---
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

print("\n--- Stationarity Test ---")
check_stationarity(data['Sold_Cars'])

# --- Step 5: ACF & PACF plots ---
plot_acf(data['Sold_Cars'].dropna(), lags=30)
plt.show()

plot_pacf(data['Sold_Cars'].dropna(), lags=30)
plt.show()

# --- Step 6: Train-test split ---
train_size = int(len(data) * 0.8)
train = data['Sold_Cars'][:train_size]
test = data['Sold_Cars'][train_size:]

# --- Step 7: SARIMA Model ---
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# --- Step 8: Make predictions ---
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

# Align indexes to avoid NaN mismatch
predictions = pd.Series(predictions, index=test.index)

# --- Step 9: RMSE ---
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\n✅ Root Mean Squared Error (RMSE): {round(rmse, 3)}')

# --- Step 10: Plot actual vs predicted ---
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Train', color='blue')
plt.plot(test.index, test, label='Actual', color='green')
plt.plot(test.index, predictions, color='red', linestyle='--', label='Predicted')
plt.xlabel('Time Index')
plt.ylabel('Number of Cars Sold')
plt.title('SARIMA Model - Cars Sold Prediction')
plt.legend()
plt.show()
```

### OUTPUT:
<img width="871" height="471" alt="image" src="https://github.com/user-attachments/assets/c638ca7c-0cc1-4165-956c-4acf75fa31a3" />
<img width="688" height="446" alt="image" src="https://github.com/user-attachments/assets/63ba0d98-832f-4bea-a8bc-aa3ccf7f2be6" />
<img width="626" height="443" alt="image" src="https://github.com/user-attachments/assets/cd24eda8-df18-46fc-a975-32e249bcb6dc" />
<img width="986" height="481" alt="image" src="https://github.com/user-attachments/assets/9b4d4cd6-b869-4808-bdd0-66e5889211d6" />




### RESULT:
Thus the program run successfully based on the SARIMA model.
