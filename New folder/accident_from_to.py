from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os
import requests
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load dataset
file_name = "AccidentsBig.csv"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Error: File '{file_name}' not found.")

df = pd.read_csv(file_name, low_memory=False)

# Filter data
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])
df = df[(df['latitude'] >= 6.5) & (df['latitude'] <= 35.5) & 
        (df['longitude'] >= 68.7) & (df['longitude'] <= 97.25)]

# Apply K-Means Clustering
coords = df[['latitude', 'longitude']].to_numpy()
if len(coords) > 0:
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)

# Function to get route using OSRM
def get_route_via_osrm(start_lat, start_lon, end_lat, end_lon):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)
    data = response.json()
    return data['routes'][0]['geometry']['coordinates']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_map', methods=['POST'])
def generate_map():
    start_location = request.form['start']
    end_location = request.form['end']
    
    geolocator = Nominatim(user_agent="route_planner")
    start = geolocator.geocode(start_location)
    end = geolocator.geocode(end_location)

    if not start or not end:
        return "Error: Unable to find one or both locations."

    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=colors[row['cluster'] % len(colors)],
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    # Get route and plot it on the map
    route = get_route_via_osrm(start.latitude, start.longitude, end.latitude, end.longitude)
    route_coordinates = [(lat, lon) for lon, lat in route]
    folium.PolyLine(route_coordinates, color='blue', weight=5, opacity=0.7).add_to(m)

    # Save map
    map_file = "templates/map.html"
    m.save(map_file)
    return render_template("map.html")

@app.route('/map')
def show_map():
    return send_file("templates/map.html")

if __name__ == '__main__':
    app.run(debug=True)

