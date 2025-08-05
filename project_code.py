#import some important libraries for EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data set
df=pd.read_excel('data of fyp.xlsx')

# first five rows of the datset
df.head()

# to check the  dataset
df.info()

#to convert the columm to numerical datatype because some alogorithm want data only fed numiric data
df['Maximum Temperature ']=pd.to_numeric(df['Maximum Temperature '],errors='coerce')
df['Minimun Temperature ']=pd.to_numeric(df['Minimun Temperature '],errors='coerce')
df['Rainfall']=pd.to_numeric(df['Rainfall'],errors='coerce')
df['water Discharge']=pd.to_numeric(df['water Discharge'],errors='coerce')

#after converting the column to numeric datatype we found  some nan values
df.isnull().sum()

# to describe the data set
df.describe()

# now we can see that there is no object type column
df.info()

#drop the row that have nan value from column which have low number of values
df=df.dropna(subset=['Maximum Temperature ','Minimun Temperature '])
df['Rainfall']=df['Rainfall'].fillna(0.05)
df['Rainfall']

#to check the shape of dataset
df.shape

# Use KNeighborsClassifier if classification
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# the rows of with out Nan values
known = df[df['water Discharge'].notna()]

# the rows that have the Nan values in water discharge
unknown = df[df['water Discharge'].isna()]

# Features to use for KNN
feature_cols = ['Maximum Temperature ', 'Minimun Temperature ', 'Average Temperature','Rainfall']
X_known = known[feature_cols]
y_known = known['water Discharge']

X_unknown = unknown[feature_cols]
scaler = StandardScaler()
X_known_scaled = scaler.fit_transform(X_known)
X_unknown_scaled = scaler.transform(X_unknown)
X_unknown_scaled

# use KNeighborsClassifier for classification
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_known_scaled, y_known)

# Predict the missing targets
y_pred = knn.predict(X_unknown_scaled)

#to add the predicted values to the original datafram where the the dataframe have the nan values
df.loc[df['water Discharge'].isna(),'water Discharge']=y_pred

df.isnull().sum()
df.head()

#EDA
import numpy as np
plt.figure(figsize=(10,5))
sns.regplot(x=df['Maximum Temperature '],y=df['water Discharge'],order=4,scatter_kws={'color':'blue'})

plt.figure(figsize=(10,5))
sns.regplot(x=df['Rainfall'],y=df['water Discharge'],scatter_kws={'color':'red'})

df['Day']=df['Date'].dt.day
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year

df=df[['Year','Month','Day','Maximum Temperature ','Minimun Temperature ','Rainfall','Average Temperature','water Discharge']]
df.head()


# This script visualizes monthly rainfall data over 10 years using a scatter plot.
import matplotlib.pyplot as plt
import calendar

# Set the figure size for the plot
plt.figure(figsize=(12, 6))

# Loop through each unique month in the dataset
for month in df['Month'].unique():
    # Filter the data for the current month
    monthly_data = df[df['Month'] == month]
    # Plot the rainfall data for the current month
    plt.scatter(monthly_data['Month'], monthly_data['Rainfall'], label=calendar.month_abbr[int(month)])

# Set x-axis ticks to abbreviated month names (Jan to Dec)
plt.xticks(ticks=range(1, 13), labels=calendar.month_abbr[1:13])

# Label x and y axes
plt.xlabel('Month')
plt.ylabel('Rainfall')

# Set the title of the plot
plt.title('Monthly Rainfall Over 10 Years')

# Display legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add a grid for better readability
plt.grid(True)

# Automatically adjust subplot params for better layout
plt.tight_layout()

# Show the final plot
plt.show()


# This script visualizes monthly water discharge trends over 10 years using a scatter plot.
# Each point represents water discharge for a specific month, labeled with abbreviated month names.

plt.figure(figsize=(12, 6))  # Set the figure size

# Loop through each unique month in the dataset
for month in df['Month'].unique():
    monthly_data = df[df['Month'] == month]  # Filter data for the current month
    plt.scatter(monthly_data['Month'], monthly_data['water Discharge'], label=calendar.month_abbr[int(month)])  # Plot data

# Set x-axis ticks to abbreviated month names (Jan to Dec)
plt.xticks(ticks=range(1,13), labels=calendar.month_abbr[1:13])

plt.xlabel('Month')  # Label for x-axis
plt.ylabel('Water Discharge')  # Label for y-axis
plt.title('Monthly Trends Over 10 Years of Water Discharge')  # Title of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Position the legend outside the plot
plt.grid(True)  # Add grid lines
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot

# This script visualizes monthly trends of maximum temperature over 10 years using a scatter plot.
# Each point represents the maximum temperature for a specific month, labeled with abbreviated month names.

plt.figure(figsize=(12, 6))  # Set the size of the figure

# Loop through each unique month in the dataset
for month in df['Month'].unique():
    monthly_data = df[df['Month'] == month]  # Filter data for the current month

    # Plot the maximum temperature for the current month
    plt.scatter(monthly_data['Month'], monthly_data['Maximum Temperature '], label=calendar.month_abbr[int(month)])

# Set x-axis ticks to month abbreviations (Jan to Dec)
plt.xticks(ticks=range(1, 13), labels=calendar.month_abbr[1:13])

plt.xlabel('Month')  # X-axis label
plt.ylabel('Maximum Temperature')  # Y-axis label
plt.title('Monthly Trends Over 10 Years')  # Title of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Show legend outside the plot
plt.grid(True)  # Enable grid lines
plt.tight_layout()  # Adjust layout to prevent label overlapping
plt.show()  # Display the plot

# This script plots the monthly trends of minimum temperature over the past 10 years using a scatter plot.
# Each month is represented with a different color label using abbreviated month names.

plt.figure(figsize=(12, 6))  # Set the size of the figure

# Loop through each unique month in the dataset
for month in df['Month'].unique():
    monthly_data = df[df['Month'] == month]  # Filter data for the current month

    # Plot the minimum temperature for the current month
    plt.scatter(monthly_data['Month'], monthly_data['Minimun Temperature '], label=calendar.month_abbr[int(month)])

# Set x-axis ticks to month abbreviations (Jan to Dec)
plt.xticks(ticks=range(1, 13), labels=calendar.month_abbr[1:13])

plt.xlabel('Month')  # Label for the x-axis
plt.ylabel('Minimum Temperature')  # Label for the y-axis
plt.title('Temperature of Past 10 Years')  # Plot title

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.grid(True)  # Add gridlines for readability
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot

df.head()

# This script plots year-wise rainfall data over the past 10 years using a scatter plot.
# Each month's data is represented as a separate series for comparison.

plt.figure(figsize=(12, 6))  # Set the size of the figure

# Loop through each unique month in the dataset
for month in df['Month'].unique():
    monthly_data = df[df['Month'] == month]  # Filter data for the current month

    # Plot rainfall values for each year for the given month
    plt.scatter(monthly_data['Year'], monthly_data['Rainfall'], label=month)

plt.xlabel('Year')  # Label for the x-axis
plt.ylabel('Rainfall')  # Label for the y-axis
plt.title('Year-wise Rainfall in the Past 10 Years')  # Title of the plot

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.grid(True)  # Show gridlines
plt.tight_layout()  # Adjust layout to avoid clipping
plt.show()  # Display the plot

# This script creates a heatmap to visualize the correlation between variables in the dataset.

plt.figure(figsize=(10, 5))  # Set the size of the heatmap figure
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # 'annot=True' displays correlation values, 'coolwarm' sets the color theme

# Creating cyclical (sin/cos) features for Month and Day to capture seasonal patterns in the data
import numpy as np

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

#no need of these column
df=df.drop(['Year','Month','Day'],axis=1)
x=df[['Maximum Temperature ','Minimun Temperature ','Average Temperature','Rainfall','Month_sin','Month_cos','Day_sin','Day_cos']]
y=df['water Discharge']

# Importing necessary libraries for model training, prediction, and evaluation using Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=160, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_

# Predictions
y_pred = best_rf_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Plotting the actual vs predicted values using Random Forest model
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.legend()
plt.title('Actual vs Predicted of random forest')
plt.show()

## XGBoost Regressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define XGBoost model
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 160],
    'max_depth': [2, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgb_regressor, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
# Predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Importing necessary libraries for Linear and Polynomial Regression
from sklearn.linear_model import LinearRegression                      # For applying Linear Regression
from sklearn.preprocessing import PolynomialFeatures                  # For generating polynomial features
from sklearn.pipeline import make_pipeline                            # For creating a pipeline of transformations
from sklearn.metrics import mean_squared_error, r2_score              # For evaluating model performance

poly=PolynomialFeatures(degree=3)
# Transform the training and testing data
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model=LinearRegression()

# Train the model
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

# Evaluate the model
print('MSE:',mean_squared_error(y_test, y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred)))
print('R²:', r2_score(y_test, y_pred))

## Support Vector Machine
from sklearn.svm import SVR

# Train an SVM with RBF kernel using GridSearchCV for best parameters
param_grid = {
    'C': [1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
    'epsilon': [0.1, 0.5, 1]  # Epsilon in the loss function
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=6, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_svr = grid_search.best_estimator_

# Make predictions
y_pred = best_svr.predict(X_test_scaled)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.legend()
plt.title('Actual vs Predicted of svm')
plt.show()

#imported required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Load and preprocess data
data = df  # Columns: rainfall, max_temp, min_temp, avg_temp, water_discharge
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 2. Create sequences with windowing
window_size = 20  # Number of past days to consider
n_features = 4    # rainfall + 3 temperature features

X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size, :-1].flatten())  # Flatten 20 days of features
    y.append(scaled_data[i + window_size, -1])             # Corresponding discharge value

X = np.array(X)
y = np.array(y)

# 3. Train-test split (time-series aware)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Build deeper ANN model
model = Sequential([
    # Input layer (window_size * n_features = 20*4 = 80)
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden layers
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),

    # Output layer
    Dense(1)
])

# 5. Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=30, restore_best_weights=True)],
    verbose=1
)

# 6. Evaluate
y_pred = model.predict(X_test)

# Inverse scaling for discharge
dummy_matrix = np.zeros((len(y_pred), scaled_data.shape[1]))
dummy_matrix[:, -1] = y_pred.flatten()
y_pred_orig = scaler.inverse_transform(dummy_matrix)[:, -1]

# True values inverse scaling
dummy_test = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_test[:, -1] = y_test
y_test_orig = scaler.inverse_transform(dummy_test)[:, -1]

# Metrics
from sklearn.metrics import r2_score, mean_squared_error
print(f"R² Score: {r2_score(y_test_orig, y_pred_orig):.4f}")
print(f"RMSE: {(mean_squared_error(y_test_orig, y_pred_orig)):.2f} m³/s")

# 7. Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_orig, label='True Discharge', linewidth=2)
plt.plot(y_pred_orig, label='Predicted Discharge', linestyle='--')
plt.title('Water Discharge Prediction (Deep ANN)\n'
          f'R²: {r2_score(y_test_orig, y_pred_orig):.3f} | '
          f'RMSE: {(mean_squared_error(y_test_orig, y_pred_orig)):.2f} m³/s')
plt.xlabel('Time Steps')
plt.ylabel('Discharge (m³/s)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# 2. Feature scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Sequence preparation
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])  # All features except target
        y.append(data[i + window_size, -1])     # Target: water discharge
    return np.array(X), np.array(y)

window_size = 20
X, y = create_sequences(scaled_data, window_size)

# 4. Train-test split (no shuffle for time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Train model
early_stop = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 7. Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse scaling
def inverse_target(scaled_y):
    dummy = np.zeros((len(scaled_y), scaled_data.shape[1]))
    dummy[:, -1] = scaled_y.flatten()
    return scaler.inverse_transform(dummy)[:, -1]

y_train_orig = inverse_target(y_train)
y_train_pred_orig = inverse_target(y_train_pred)
y_test_orig = inverse_target(y_test)
y_test_pred_orig = inverse_target(y_test_pred)

# 8. Evaluate
print("\n📊 Training Results")
print("R² Score:", r2_score(y_train_orig, y_train_pred_orig))
print("RMSE:", mean_squared_error(y_train_orig, y_train_pred_orig))

print("\n📊 Testing Results")
print("R² Score:", r2_score(y_test_orig, y_test_pred_orig))
print("RMSE:", mean_squared_error(y_test_orig, y_test_pred_orig))

# 10. Metrics for bar graph
train_r2 = r2_score(y_train_orig, y_train_pred_orig)
test_r2 = r2_score(y_test_orig, y_test_pred_orig)

train_rmse = mean_squared_error(y_train_orig, y_train_pred_orig)
test_rmse = mean_squared_error(y_test_orig, y_test_pred_orig)

# 11. Plotting bar graphs
plt.figure(figsize=(10, 5))

# R² Score comparison
plt.subplot(1, 2, 1)
plt.bar(['Train', 'Test'], [train_r2, test_r2], color=['skyblue', 'salmon'])
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)

# RMSE comparison
plt.subplot(1, 2, 2)
plt.bar(['Train', 'Test'], [train_rmse, test_rmse], color=['skyblue', 'salmon'])
plt.title('RMSE Comparison')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()




