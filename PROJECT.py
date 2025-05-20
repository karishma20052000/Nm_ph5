# Install necessary libraries
!pip install -q numpy pandas matplotlib seaborn scikit-learn plotly
# Import Python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
# Simulate sensor data for waste-to-energy plant
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'temperature': np.random.normal(800, 50, n),  # Reactor temperature (°C)
    'pressure': np.random.normal(5, 0.5, n),      # Pressure in atm
    'flow_rate': np.random.normal(20, 3, n),      # Syngas flow rate (m³/hr)
    'waste_input': np.random.normal(100, 15, n)   # Waste input (kg/hr)
})

# Simulate energy output (dependent variable)
data['energy_output'] = (data['waste_input'] * 0.4) + (data['temperature'] * 0.1) - (data['pressure'] * 2) + np.random.normal(0, 10, n)

# Show first few rows
data.head()
# Visualize feature relationships
sns.pairplot(data)
plt.show()# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.title('Feature Correlation Heatmap')
plt.show()
# Train-test split
X = data[['temperature', 'pressure', 'flow_rate', 'waste_input']]
y = data['energy_output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Output evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")
# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Energy Output")
plt.ylabel("Predicted Energy Output")
plt.title("Actual vs Predicted Energy Output")
plt.grid(True)
plt.show()
# 3D visualization using Plotly
fig = px.scatter_3d(data, x='temperature', y='waste_input', z='energy_output',
                    color='flow_rate', title='3D Visualization of Energy Output')
fig.show()

