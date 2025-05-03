import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os
import requests
from geopy.geocoders import Nominatim
import sys  # <-- Import sys to read command-line arguments

# Ensure correct usage
if len(sys.argv) != 3:
    print("Usage: python from_to.py <start_location> <end_location>")
    sys.exit(1)

start_location = sys.argv[1]
end_location = sys.argv[2]

# Load dataset and check if the file exists
file_name = "AccidentsBig.csv"
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found. Please check the file path.")
    sys.exit(1)

# Read the dataset
df = pd.read_csv(file_name, low_memory=False)
required_columns = ['Accident_Index', 'latitude', 'longitude', 'Number_of_Casualties']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns {missing_columns} in the dataset.")
    sys.exit(1)

# Convert latitude and longitude to numeric values
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop NaN and filter invalid values
df = df.dropna(subset=['latitude', 'longitude'])
df = df[(df['latitude'].between(6.5, 35.5)) & (df['longitude'].between(68.7, 97.25))]

# Check remaining data points
if df.empty:
    print("Error: No valid accident data found after filtering.")
    sys.exit(1)

# Apply K-Means clustering
from sklearn.cluster import KMeans
coords = df[['latitude', 'longitude']].to_numpy()
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(coords)

# Create the map
center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)

colors = ['red', 'blue', 'green', 'purple', 'orange']

# Add accident data markers
for _, row in df.iterrows():
    popup_text = (
        f"Accident Index: {row['Accident_Index']}<br>"
        f"Location: ({row['latitude']}, {row['longitude']})<br>"
        f"Casualties: {row['Number_of_Casualties']}"
    )
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=colors[row['cluster'] % len(colors)],
        fill=True,
        fill_color=colors[row['cluster'] % len(colors)],
        fill_opacity=0.7,
        popup=popup_text
    ).add_to(marker_cluster)

# Get route between start and end locations
def get_route(start_lat, start_lon, end_lat, end_lon):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)
    data = response.json()

    if "routes" not in data or not data["routes"]:
        print("Error: Could not retrieve route.")
        sys.exit(1)

    return data['routes'][0]['geometry']['coordinates']

# Get coordinates using Geopy
geolocator = Nominatim(user_agent="route_planner")
start = geolocator.geocode(start_location)
end = geolocator.geocode(end_location)

if not start or not end:
    print("Error: Could not find specified locations.")
    sys.exit(1)

# Fetch and plot the route
route = get_route(start.latitude, start.longitude, end.latitude, end.longitude)
route_coordinates = [(lat, lon) for lon, lat in route]
folium.PolyLine(route_coordinates, color='blue', weight=5, opacity=0.7).add_to(m)

# Save the map in the static folder
map_file = "static/accident_clusters_with_route.html"
m.save(map_file)
print("Map successfully generated!")
