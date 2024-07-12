import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'temperature_data.csv'
df = pd.read_csv(file_path)

# Step 2: Basic Data Exploration
# df.head() pratique pour les fichiers très lourds où on peut pas trop voir les données; print les 5 premieres colonnes
print("First few rows of the dataset:")
print(df.head())

# df.describe() donne les stats de base pour la temperature
print("\nSummary statistics of the dataset:")
print(df.describe())

# Check for missing values
# pratique pour filtrer des erreurs dans le dataset 
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Step 3: Data Cleaning
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows with missing values (if any)
df = df.dropna()

# Step 4: Data Visualization
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Temperature'], marker='o')
plt.title('Temperature Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

# Step 5: Calculate Basic Statistics
mean_temp = df['Temperature'].mean()
median_temp = df['Temperature'].median()
std_temp = df['Temperature'].std()

print(f"\nMean Temperature: {mean_temp:.2f} °C")
print(f"Median Temperature: {median_temp:.2f} °C")
print(f"Standard Deviation of Temperature: {std_temp:.2f} °C")