import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
import os

# Load dataset and check if the file exists
file_name = "AccidentsBig.csv"

if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found. Please check the file path.")
    exit()

# Read the dataset (handle mixed data types)
print("Loading the dataset...")
df = pd.read_csv(file_name, low_memory=False)
print(f"Dataset loaded. Number of rows: {len(df)}")

# Check if necessary columns exist
required_columns = ['Accident_Index', 'latitude', 'longitude', 'Number_of_Casualties']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns {missing_columns} in the dataset.")
    exit()

# Convert latitude and longitude to numeric values
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop rows with NaN latitude or longitude
df = df.dropna(subset=['latitude', 'longitude'])

# Filter out invalid values for latitude and longitude
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90) & 
        (df['longitude'] >= -180) & (df['longitude'] <= 180)]

# Filter for locations within India
df = df[(df['latitude'] >= 6.5) & (df['latitude'] <= 35.5) & 
        (df['longitude'] >= 68.7) & (df['longitude'] <= 97.25)]

# Check remaining rows
print(f"Remaining rows after filtering: {len(df)}")

# Extract coordinates for clustering
coords = df[['latitude', 'longitude']].to_numpy()

# Ensure no infinite values
coords = coords[~np.isinf(coords).any(axis=1)]

# Check for valid coordinates
if len(coords) == 0:
    print("Error: No valid data points found for clustering after cleaning.")
    exit()

# Apply K-Means clustering (adjust number of clusters as needed)
print("Applying K-Means clustering...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coords)
print("Clustering complete.")

# Create an interactive map
center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

# Initialize MarkerCluster for better performance
marker_cluster = MarkerCluster().add_to(m)

# Define cluster colors
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Add clustered markers with Accident Index and Casualties
print("Adding markers to the map...")

for _, row in df.iterrows():
    accident_index = row['Accident_Index']
    casualties = row['Number_of_Casualties']
    popup_text = (
        f"Accident Index: {accident_index}<br>"
        f"Location: ({row['latitude']}, {row['longitude']})<br>"
        f"Casualties: {casualties}"
    )
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,  # Slightly increased for better visibility
        color=colors[row['cluster'] % len(colors)],
        fill=True,
        fill_color=colors[row['cluster'] % len(colors)],
        fill_opacity=0.7,
        popup=popup_text  # Display accident details
    ).add_to(marker_cluster)

# Save map to an HTML file
map_file = "accident_clusters_india.html"
m.save(map_file)
print(f"Map saved as '{map_file}'. Open in a browser to view.")
