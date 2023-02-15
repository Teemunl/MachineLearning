import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.dates as mdates

# Read the historical data of Bitcoin prices
df = pd.read_csv("BTC-USD.csv", parse_dates=["Date"])

# Convert the date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Convert the date to ordinal format using pandas.Timestamp.toordinal method
X = df["Date"].apply(pd.Timestamp.toordinal).values.reshape(-1, 1)

# Format the date column as a string without the time part
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

# Filter the data for 2023
df = df[["Date", "Close"]]

# Use the closing price as the target variable
y = df["Close"].values

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions for 2023
X_pred = np.arange(X.min(), X.max() + 1).reshape(-1, 1)
y_pred = model.predict(X_pred)

# Make predictions for the rest of the year after February
X_pred_future = np.arange(X.max() + 1, X.max() + 305).reshape(-1, 1)
y_pred_future = model.predict(X_pred_future)

# Flatten the arrays using np.ravel
X_pred_future = np.ravel(X_pred_future)
y_pred_future = np.ravel(y_pred_future)

# Convert the ordinal values of X_pred_future back to datetime format
X_pred_future = [datetime.date.fromordinal(int(x)) for x in X_pred_future]

# Append the predicted values to the original data frame
df_pred = pd.DataFrame({"Date": X_pred_future, "Close": y_pred_future})

# Add the other columns to the data frame using the reindex method
# Use the original data frame's columns as the new columns
# Fill the missing values with NaN
df_pred = df_pred.reindex(columns=df.columns, fill_value=np.nan)

df_all = pd.concat([df, df_pred])

# Save the data frame as a new csv file
df_all.to_csv("BTC-USD-2023.csv", index=False)

# Read the new csv file and plot the graph as before
df_new = pd.read_csv("BTC-USD-2023.csv")
df_new["Date"] = pd.to_datetime(df_new["Date"])
y_new = df_new["Close"].values
X_new = df_new["Date"].apply(pd.Timestamp.toordinal).values.reshape(-1, 1)
plt.plot(X_new, y_new, label="Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Bitcoin Price Prediction in 2023")
plt.legend()

# Create a DateFormatter object with the format "%B" for the full name of the month
monthFmt = mdates.DateFormatter("%B")

# Set the DateFormatter object as the major formatter for the x-axis
plt.gca().xaxis.set_major_formatter(monthFmt)

# Create a MonthLocator object with the interval 1 for every month
monthLoc = mdates.MonthLocator(interval=1)

# Set the MonthLocator object as the major locator for the x-axis
plt.gca().xaxis.set_major_locator(monthLoc)

# Create a tuple of two datetime values, one for the start date of 2023 and one for the end date of 2023
x_lim = (datetime.date(2023, 1, 1), datetime.date(2023, 12, 31))

# Set the tuple as the limit of the x-axis
plt.gca().set_xlim(x_lim)

# Convert the ordinal values of X, X_pred and X_pred_future back to datetime format
X = [datetime.date.fromordinal(int(x)) for x in X]
X_pred = [datetime.date.fromordinal(int(x)) for x in X_pred]

plt.plot(X, y, label="Historical Price") 
plt.plot(X_pred_future, y_pred_future, label="Predicted Price")
# Show the graph
plt.show()