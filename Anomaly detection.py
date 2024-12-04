import random
import pandas as pd

# Generate normal data
normal_data = [random.uniform(20, 30) for _ in range(100)]

# Generate some anomalies
high_anomalies = [random.uniform(50, 60) for _ in range(5)]
low_anomalies = [random.uniform(5, 10) for _ in range(5)]
anomalies = high_anomalies + low_anomalies

# Combine normal data and anomalies
data = normal_data + anomalies
random.shuffle(data)  # Shuffle the data to mix normal and anomalies

# Save to a CSV file
df = pd.DataFrame({'Temperature': data})
df.to_csv('temperature_data.csv', index=False)
print("Dataset created and saved as 'temperature_data.csv'!")

import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('temperature_data.csv')

# Plot the temperature data
plt.figure(figsize=(10, 5))
plt.plot(df['Temperature'], label='Temperature Readings', color='blue')
plt.axhline(30, color='green', linestyle='--', label='Normal Upper Limit')
plt.axhline(20, color='green', linestyle='--', label='Normal Lower Limit')
plt.title("Temperature Readings")
plt.xlabel("Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()


from sklearn.ensemble import IsolationForest

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)  # 10% contamination assumed

# Fit the model and predict anomalies
df['Anomaly'] = model.fit_predict(df[['Temperature']])

# Anomalies are labeled as -1; normal points are labeled as 1
anomalies = df[df['Anomaly'] == -1]
normal = df[df['Anomaly'] == 1]

print("Number of detected anomalies:", len(anomalies))
print(anomalies)


# Plot temperature data with anomalies highlighted
plt.figure(figsize=(10, 5))
plt.plot(df['Temperature'], label='Temperature Readings', color='blue')
plt.scatter(anomalies.index, anomalies['Temperature'], color='red', label='Detected Anomalies')
plt.axhline(30, color='green', linestyle='--', label='Normal Upper Limit')
plt.axhline(20, color='green', linestyle='--', label='Normal Lower Limit')
plt.title("Anomaly Detection in Temperature Readings")
plt.xlabel("Index")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()
