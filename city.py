import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
from geopy.geocoders import Nominatim
import folium

# Function to convert location name to latitude and longitude
def get_lat_long_from_place(place_name):
    geolocator = Nominatim(user_agent="accident_prediction")
    location = geolocator.geocode(place_name)
    if location:
        return location.latitude, location.longitude
    else:
        print(f"Could not find the location: {place_name}")
        return None, None

# Load dataset (update the path if needed)
file_path = 'AccidentsBig.csv'  # Use your file path
data = pd.read_csv(file_path)

# Drop missing values
data.dropna(subset=['latitude', 'longitude', 'Accident_Severity', 'Time'], inplace=True)

# Handle the case where 'Time' is in 'HH:MM' format
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.strftime('%H:%M')

# Convert 'Time' to hour and minute
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
data['Minute'] = pd.to_datetime(data['Time'], format='%H:%M').dt.minute

# Select features and target
X = data[['latitude', 'longitude', 'Hour', 'Minute']]
y = data['Accident_Severity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering to group accident-prone areas
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
data['Cluster_Labels'] = kmeans.fit_predict(X_scaled)

# Add cluster labels as a feature
X = data[['latitude', 'longitude', 'Hour', 'Minute', 'Cluster_Labels']]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Check accuracy
print("Random Forest Training Accuracy:", rf_model.score(X_train, y_train))
print("Random Forest Testing Accuracy:", rf_model.score(X_test, y_test))

# Predict accident severity based on location name
def predict_accident_severity(place_name):
    # Get latitude and longitude for the place name
    latitude, longitude = get_lat_long_from_place(place_name)
    
    if latitude is None or longitude is None:
        return
    
    print(f"Location - Latitude: {latitude}, Longitude: {longitude}")

    # Get the current time
    current_time = datetime.datetime.now()
    user_hour = current_time.hour
    user_minute = current_time.minute

    # Predict cluster label for the location
    user_cluster = kmeans.predict(scaler.transform([[latitude, longitude, user_hour, user_minute]]))[0]

    # Prepare input data for prediction
    input_data = [[latitude, longitude, user_hour, user_minute, user_cluster]]
    probability = rf_model.predict_proba(input_data)[0]  # Extract first element

    # Format probabilities as percentages
    formatted_prob = f"Low Severity: {probability[0]*100:.2f}% | Medium Severity: {probability[1]*100:.2f}% | High Severity: {probability[2]*100:.2f}%"
    print(f"Prediction for {place_name} (Lat: {latitude}, Long: {longitude}):")
    print(formatted_prob)

    return formatted_prob, latitude, longitude

# Visualize on the same HTML page
def visualize_on_map(place_name, latitude, longitude, prediction):
    # Create a map centered on the given location
    map_location = folium.Map(location=[latitude, longitude], zoom_start=14)

    # Add a marker for the location with the prediction
    folium.Marker([latitude, longitude], popup=f"Accident Severity Prediction: {prediction}").add_to(map_location)

    # Save map as an HTML file and add prediction text to the map popup
    map_location.save(f'city_accident_map.html')
    print(f"Map saved as 'city_accident_map.html'. Open it in your browser.")

# Example: Get place name from the user at runtime
if __name__ == "__main__":
    place_name = input("Enter the location name (e.g., City, Address): ").strip()
    
    # Predict the accident severity at the given location
    prediction, latitude, longitude = predict_accident_severity(place_name)
    
    if prediction:
        # Visualize the place on the map with the prediction
        visualize_on_map(place_name, latitude, longitude, prediction)
