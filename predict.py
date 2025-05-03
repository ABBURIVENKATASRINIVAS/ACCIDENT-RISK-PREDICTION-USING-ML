import joblib
import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np

# Load the trained model and scaler
rf_model, _, _, _, scaler = joblib.load("accident_severity_model.pkl")

# Function to convert location name to coordinates
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        print("⚠️ Location not found! Using default coordinates.")
        return 0, 0  # Default coordinates if location is invalid

# Function to predict accident severity
def predict_accident_severity(weather, road_surface, light, location_name):
    try:
        # Convert inputs to integers
        weather = int(weather)
        road_surface = int(road_surface)
        light = int(light)
    except ValueError:
        return "⚠️ Please enter valid numeric values for Weather, Road Surface, and Light Conditions!"

    # Validate input values
    valid_weather = {2, 1, 9, 3, 8, 4, 5, 7, 6}
    valid_road_surface = {2, 1, 4, 3, 5, -1}
    valid_light = {1, 4, 7, 5, 6}

    if weather not in valid_weather or road_surface not in valid_road_surface or light not in valid_light:
        return "⚠️ Invalid input values!"

    # Convert location name to coordinates
    latitude, longitude = get_coordinates(location_name)

    # Scale numerical inputs
    scaled_features = scaler.transform([[longitude, latitude]])

    # Create input array
    input_data = np.array([[weather, road_surface, light, scaled_features[0][0], scaled_features[0][1]]])

    # Predict accident severity
    prediction = rf_model.predict(input_data)[0]

    return f"🚨 Predicted Accident Severity: {prediction}"

# If running this script directly, take user input
if __name__ == "__main__":
    weather = input("Enter Weather Conditions (2, 1, 9, 3, 8, 4, 5, 7, 6): ")
    road_surface = input("Enter Road Surface Conditions (2, 1, 4, 3, 5, -1): ")
    light = input("Enter Light Conditions (1, 4, 7, 5, 6): ")
    location_name = input("Enter Location Name: ")

    # Predict and print result
    print(predict_accident_severity(weather, road_surface, light, location_name))
