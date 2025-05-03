from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import subprocess
from weather import Weather, WeatherException
import os
import folium
import subprocess

from city import predict_accident_severity
app = Flask(__name__)
app.config.from_pyfile('config/config.cfg')
w = Weather(app.config)

# Load the ML model safely
model_path = 'test1.pkl'
if os.path.exists(model_path):
    test = pickle.load(open(model_path, 'rb'))
else:
    test = None
    print("Warning: ML model file 'test1.pkl' not found.")

@app.route('/')
def home():
    return render_template("gmap.html")

@app.route('/show')
def show_input():
    return render_template("t.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if test is None:
        return render_template('t.html', pred="Error: ML model is missing.")

    try:
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        print(int_features)
        print(final)
        prediction = test.predict(final)

        severity = "Minor" if prediction == 0 else "Major"
        return render_template('t.html', pred=f"\t\t\t\t\tProbability of accident severity is: {severity}")
    
    except Exception as e:
        return render_template('t.html', pred=f"Error in prediction: {str(e)}")

@app.route('/Map')
def map1():
    return render_template("accident_clusters_india.html")

@app.route('/Graphs')
def graph():
    return render_template("graph.html")

@app.route('/Map1')
def map2():
    return render_template("ur.html")

@app.route('/Map2')
def map3():
    return render_template("bs.html")

@app.route('/Map3')
def map4():
    return render_template("hm.html")

@app.route('/Pie')
def pie():
    return render_template("pie.html")

@app.route('/wf')
def wf():
    return render_template("wf.html")

@app.route('/result', methods=['POST', 'GET'])
def result_page():
    if request.method == 'POST':
        location = request.form
        w.set_location(location.get('location'))

        try:
            return render_template('result.html', data=w.get_forecast_data())
        except WeatherException:
            app.logger.error("WeatherException occurred")
            return render_template('error.html')
    else:
        return redirect(url_for('show_input'))


@app.route('/city_prediction')
def city_prediction_page():
    return render_template("city_prediction.html")


@app.route('/run_from_to', methods=['POST'])
def run_from_to():
    start_location = request.form.get('start_location')
    end_location = request.form.get('end_location')

    if not start_location or not end_location:
        return "Error: Start and End locations are required", 400

    try:
        # Run from_to.py with arguments
        process = subprocess.run(
            ['python', 'from_to.py', start_location, end_location], 
            capture_output=True, text=True
        )

        if process.returncode != 0:
            return f"Error running script: {process.stderr}", 500

        # Redirect to generated map
        return redirect('/static/accident_clusters_with_route.html')

    except Exception as e:
        return f"Internal Server Error: {str(e)}", 500
@app.route('/predict_city', methods=['POST'])
def predict_city():
    place_name = request.form.get('location')

    if not place_name:
        return render_template('city_prediction.html', pred_city="Error: Please enter a location.", map_path=None)

    try:
        # Predict accident severity
        prediction, latitude, longitude = predict_accident_severity(place_name)

        if prediction:
            map_path = visualize_on_map(place_name, latitude, longitude, prediction)  # Generate map
            return render_template('city_prediction.html', pred_city=f"Prediction for {place_name}: {prediction}", map_path=map_path)
        else:
            return render_template('city_prediction.html', pred_city="Error: Unable to determine location.", map_path=None)
    
    except Exception as e:
        return render_template('city_prediction.html', pred_city=f"Error: {str(e)}", map_path=None)
def visualize_on_map(place_name, latitude, longitude, prediction):
    map_location = folium.Map(location=[latitude, longitude], zoom_start=14)

    # Add a marker for the location with the prediction
    folium.Marker(
        [latitude, longitude], 
        popup=f"Prediction: {prediction}",
        icon=folium.Icon(color="red" if "High" in prediction else "green")
    ).add_to(map_location)

    # Define the path for the saved map
    map_path = "static/city_accident_map.html"
    
    # Save the map
    map_location.save(map_path)
    
    return map_path
@app.route('/route_input')
def route_input():
    return render_template("route_input.html")

@app.route('/from_to', methods=['POST'])
def from_to():
    start_location = request.form.get('start_location')
    end_location = request.form.get('end_location')

    if not start_location or not end_location:
        return "Error: Start and End locations are required", 400

    try:
        # Run from_to.py with arguments
        process = subprocess.run(
            ['python', 'from_to.py', start_location, end_location], 
            capture_output=True, text=True
        )

        if process.returncode != 0:
            return f"Error running script: {process.stderr}", 500

        # Redirect to the generated map
        return render_template('from_to.html', map_path="/static/accident_clusters_with_route.html")

    except Exception as e:
        return f"Internal Server Error: {str(e)}", 500

@app.route('/severity_input')
def severity_input():
    return render_template("predict_input.html")

import joblib
import pandas as pd
from flask import Flask, render_template, request
from geopy.geocoders import Nominatim  # Ensure geopy is correctly imported
import numpy as np



# Load the trained model and scaler
model_data = joblib.load("accident_severity_model.pkl")
rf_model = model_data[0]  # Extract the RandomForest model
scaler = model_data[4]  # Extract the scaler (the 5th item in the saved tuple)

# Function to convert location name to coordinates
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        print("⚠️ Location not found! Using default coordinates.")
        return 0, 0  # Default coordinates if location is invalid

# Route to handle prediction form submission
@app.route('/predict_custom', methods=['POST', 'GET'])
def predict_custom():
    if request.method == 'POST':
        try:
            # Get input data from the form submission
            weather = request.form.get('weather')
            road_surface = request.form.get('road')
            light = request.form.get('light')
            location_name = request.form.get('location')

            # Convert inputs to integers
            weather = int(weather)
            road_surface = int(road_surface)
            light = int(light)

            # Validate input values
            valid_weather = {2, 1, 9, 3, 8, 4, 5, 7, 6}
            valid_road_surface = {2, 1, 4, 3, 5, -1}
            valid_light = {1, 4, 7, 5, 6}

            if weather not in valid_weather or road_surface not in valid_road_surface or light not in valid_light:
                return render_template('predict_result.html', pred="⚠️ Invalid input values!")

            # Convert location name to coordinates
            latitude, longitude = get_coordinates(location_name)

            # Scale numerical features
            scaled_features = scaler.transform([[longitude, latitude]])

            # Create input array
            input_data = np.array([[weather, road_surface, light, scaled_features[0][0], scaled_features[0][1]]])

            # Predict accident severity
            prediction = rf_model.predict(input_data)[0]

            # Map prediction to severity values
            if prediction == 1:
                severity = "Severity: Minor (1)"
            elif prediction == 2:
                severity = "Severity: Moderate (2)"
            elif prediction == 3:
                severity = "Severity: Major (3)"
            else:
                severity = "Unknown severity"

            # Return the result in the same route
            return render_template('predict_result.html', pred=severity)

        except Exception as e:
            return render_template('predict_result.html', pred=f"Error during prediction: {str(e)}")
            
            
    return render_template('predict_input.html')



if __name__ == "__main__":
    app.run(debug=True)
