# Cedar Creek Dairy Dataset - Full Analysis & Modeling
# Includes Static EDA, Interactive EDA, and Predictive Modeling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv('cedar_creek_dairy_dataset_profit.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)

# -----------------------------
# STATIC EDA
# -----------------------------
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe())

numeric_cols = ['Milk_Yield_kg', 'Feed_Intake_kg', 'Health_Score', 'Revenue_USD', 'Feed_Cost_USD', 'Profit_USD']
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# -----------------------------
# INTERACTIVE EDA
# -----------------------------
fig = px.histogram(df, x='Milk_Yield_kg', nbins=30, title='Milk Yield Distribution', marginal='box')
fig.show()

corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Heatmap')
fig.show()

monthly_summary = df.groupby('Month')[['Revenue_USD', 'Profit_USD']].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly_summary['Month'], y=monthly_summary['Revenue_USD'], mode='lines+markers', name='Revenue'))
fig.add_trace(go.Scatter(x=monthly_summary['Month'], y=monthly_summary['Profit_USD'], mode='lines+markers', name='Profit'))
fig.update_layout(title='Monthly Revenue vs Profit', xaxis_title='Month', yaxis_title='USD')
fig.show()

# -----------------------------
# PREDICTIVE MODELING
# -----------------------------
# Target: Profit_USD
X = df.drop(columns=['Profit_USD', 'Date', 'Month', 'Cow_ID'])
y = df['Profit_USD']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nLinear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R²:", r2_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))

# Feature Importance for Random Forest
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[:10], y=importances.index[:10])
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()