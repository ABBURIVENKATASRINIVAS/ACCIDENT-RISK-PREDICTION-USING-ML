import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import datetime
from geopy.geocoders import Nominatim

# Initialize geocoder
geolocator = Nominatim(user_agent="accident_prediction")

# Function to get coordinates from location name
def get_coordinates(location_name):
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        print(f"Location {location_name} not found.")
        return None, None

# Load additional data (weather, traffic, etc.)
def get_additional_features(location, time):
    # Placeholder function to get additional features
    # You can call weather APIs, traffic data, etc. here
    weather_data = get_weather_data(location)
    traffic_data = get_traffic_data(location, time)
    return weather_data + traffic_data  # Concatenate all data

def get_weather_data(location):
    # Placeholder function for weather data
    # This function should return weather data like temperature, visibility, rainfall, etc.
    return [15, 0.5]  # e.g., 15°C, 0.5 mm of rain

def get_traffic_data(location, time):
    # Placeholder function for traffic data
    # This function should return traffic data like traffic density, vehicle speeds, etc.
    return [300, 60]  # e.g., 300 vehicles on the road, average speed 60 km/h

# Load dataset
data = pd.read_csv('AccidentsBig.csv')

# Drop missing values
data.dropna(subset=['latitude', 'longitude', 'Accident_Severity', 'Time'], inplace=True)

# Convert time to hour and minute
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
data['Minute'] = pd.to_datetime(data['Time'], format='%H:%M').dt.minute

# Feature extraction (add your additional features here)
additional_features = [get_additional_features(row['latitude'], row['Time']) for index, row in data.iterrows()]
data['Weather_Temperature'], data['Rainfall'] = zip(*[f[:2] for f in additional_features])  # Extract weather data
data['Traffic_Density'], data['Speed'] = zip(*[f[2:] for f in additional_features])  # Extract traffic data

# Select features and target
X = data[['latitude', 'longitude', 'Hour', 'Minute', 'Weather_Temperature', 'Rainfall', 'Traffic_Density', 'Speed']]
y = data['Accident_Severity']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Check Accuracy
print("Training Accuracy:", rf_model.score(X_train, y_train))
print("Testing Accuracy:", rf_model.score(X_test, y_test))

# User input for location
location_name = input("Enter the location name (e.g., 'Bangalore'): ")
latitude, longitude = get_coordinates(location_name)

if latitude and longitude:
    # Get current time
    time = datetime.datetime.now().strftime('%H:%M')

    # Extract additional features (weather, traffic)
    additional_data = get_additional_features([latitude, longitude], time)
    user_data = [latitude, longitude, int(time.split(':')[0]), int(time.split(':')[1])] + additional_data

    # Predict accident probability
    probability = rf_model.predict_proba([user_data])[0]

    # Print results
    print(f"Accident Occurrence Probability: {probability[1] * 100:.2f}%")
else:
    print("Unable to get coordinates for the specified location.")
