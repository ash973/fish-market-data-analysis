import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import opendatasets as od
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

od.download("https://www.kaggle.com/datasets/vipullrathod/fish-market/data")

dataset = pd.read_csv("fish-market/Fish.csv")

#  Scatter plot (Species vs Length1)
alt.Chart(dataset).mark_circle().encode(
    x='Species',
    y='Length1'
).interactive()

#  Add noise to 'Weight' and evaluate regression performance
noise_level = 100
noise = np.random.normal(0, noise_level, size=len(dataset))
dataset['Weight_noisy'] = dataset['Weight'] + noise

# Regression with original data
X = dataset[['Length1']]
y = dataset['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Mean Squared Error (Original Data): {mean_squared_error(y_test, y_pred):.2f}")

# Regression with noisy data
X_noisy = dataset[['Length1']]
y_noisy = dataset['Weight_noisy']
X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

model_noisy = LinearRegression()
model_noisy.fit(X_noisy_train, y_noisy_train)
y_noisy_pred = model_noisy.predict(X_noisy_test)

print(f"Mean Squared Error (Noisy Data): {mean_squared_error(y_noisy_test, y_noisy_pred):.2f}")
print(f"R-squared (Original Data): {r2_score(y_test, y_pred):.2f}")
print(f"R-squared (Noisy Data): {r2_score(y_noisy_test, y_noisy_pred):.2f}")

# Visualize noisy data
alt.Chart(dataset).mark_circle().encode(
    x='Length1',
    y='Weight_noisy'
).interactive()

#  Scatter plots of all parameters vs Species
parameters = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
colors = ['red', 'green', 'blue', 'orange', 'purple']

plt.figure(figsize=(12, 8))
for i, param in enumerate(parameters):
    plt.scatter(dataset['Species'], dataset[param], color=colors[i], label=param, alpha=0.7)

plt.xlabel('Species')
plt.ylabel('Parameters')
plt.title('Scatter Plot of Species vs. Parameters')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for param in parameters:
    noise = np.random.normal(0, 10, size=len(dataset))
    dataset[param + '_noisy'] = dataset[param] + abs(noise)

for param in parameters:
    X = dataset[[param]]
    y = dataset['Weight']
    X_noisy = dataset[[param + '_noisy']]
    y_noisy = dataset['Weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_noisy = LinearRegression()
    model_noisy.fit(X_noisy_train, y_noisy_train)
    y_noisy_pred = model_noisy.predict(X_noisy_test)

    print(f"\nParameter: {param}")
    print(f"  MSE (Original): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"  R-squared (Original): {r2_score(y_test, y_pred):.2f}")
    print(f"  MSE (Noisy): {mean_squared_error(y_noisy_test, y_noisy_pred):.2f}")
    print(f"  R-squared (Noisy): {r2_score(y_noisy_test, y_noisy_pred):.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, label='Original Data', alpha=0.7)
    plt.scatter(X_noisy_test, y_noisy_test, label='Noisy Data', alpha=0.7)
    plt.plot(X_test, y_pred, color='red', label='Regression Line (Original)')
    plt.plot(X_noisy_test, y_noisy_pred, color='green', label='Regression Line (Noisy)')
    plt.xlabel(param)
    plt.ylabel('Weight')
    plt.title(f'{param} vs Weight: Original vs Noisy')
    plt.legend()
    plt.show()
