import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "data_sets/EC-2/non_smart_meter_data.csv"  # Replace with your dataset file path
try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 1: Data Cleaning and Preprocessing
print("Dataset preview:\n", df.head())

# Convert the 'consumption' column to numeric
if 'consumption' in df.columns:
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')

# Exclude non-numeric columns for filling missing values
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Feature Engineering
# Add example calculated features (adjust column names to fit your dataset)
if 'consumption' in df.columns:
    df['PeakUsage'] = df['consumption'] * 0.7  # Example: Assume 70% of usage is during peak hours
    df['OffPeakUsage'] = df['consumption'] * 0.3

# Step 2: Cluster Analysis for Inefficient Energy Usage
print("\nClustering households based on energy consumption patterns...")
features = df[['consumption', 'PeakUsage', 'OffPeakUsage']]  # Adjust as needed
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Evaluate clustering
silhouette_avg = silhouette_score(features, df['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# Step 3: Visualizations
print("\nVisualizing clusters...")
sns.scatterplot(data=df, x='consumption', y='PeakUsage', hue='Cluster', palette='viridis')
plt.title('Energy Consumption Clusters')
plt.xlabel('Total Consumption')
plt.ylabel('Peak Usage')
plt.show()
