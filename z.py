import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime
import geocoder
import folium

# Function to get user location (IP-based geolocation)
def get_location():
    try:
        location = geocoder.ip('me')  # Get location based on IP
        if location.ok:
            lat = location.latlng[0]  # Latitude
            lng = location.latlng[1]  # Longitude
            return lat, lng
        else:
            print("Unable to retrieve location.")
            return None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None

# Load dataset (update the path if needed)
file_path = 'AccidentsBig.csv'  # Use your file path
data = pd.read_csv(file_path)

# Drop missing values
data.dropna(subset=['latitude', 'longitude', 'Accident_Severity', 'Time'], inplace=True)

# Convert 'Time' to hour and minute
data['Hour'] = pd.to_datetime(data['Time']).dt.hour
data['Minute'] = pd.to_datetime(data['Time']).dt.minute

# Select features and target
X = data[['latitude', 'longitude', 'Hour', 'Minute']]
y = data['Accident_Severity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster_Labels'] = kmeans.fit_predict(X_scaled)

# Add cluster labels as a feature
X = data[['latitude', 'longitude', 'Hour', 'Minute', 'Cluster_Labels']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Check Accuracy
print("Random Forest Training Accuracy:", rf_model.score(X_train, y_train))
print("Random Forest Testing Accuracy:", rf_model.score(X_test, y_test))

# Predict accident probability at current location and time
if __name__ == "__main__":
    coordinates = get_location()  # Get current location using IP-based geolocation
    if coordinates:
        user_latitude, user_longitude = coordinates
        print(f"Current Location - Latitude: {user_latitude}, Longitude: {user_longitude}")

        # Get current time
        current_time = datetime.datetime.now()
        user_hour = current_time.hour
        user_minute = current_time.minute

        # Predict cluster label
        user_cluster = kmeans.predict(scaler.transform([[user_latitude, user_longitude, user_hour, user_minute]]))[0]

        # Prepare input data for prediction
        input_data = [[user_latitude, user_longitude, user_hour, user_minute, user_cluster]]
        probability = rf_model.predict_proba(input_data)[0]  # Extract first element

        # Format probabilities as percentages
        formatted_prob = f"Low Severity: {probability[0]*100:.2f}% | Medium Severity: {probability[1]*100:.2f}% | High Severity: {probability[2]*100:.2f}%"
        print("Probability of accident at current coordinates and time:", formatted_prob)

        # Plot the current location on the map using Folium
        map_location = folium.Map(location=[user_latitude, user_longitude], zoom_start=14)
        
        # Add a marker for the current location
        folium.Marker([user_latitude, user_longitude], 
                      popup=f"Accident Probability:\n{formatted_prob}").add_to(map_location)

        # Save map as an HTML file
        map_location.save('current_location_map.html')
        print("Map saved as 'current_location_map.html'. Open it in your browser.")
    else:
        print("Failed to retrieve location.")
