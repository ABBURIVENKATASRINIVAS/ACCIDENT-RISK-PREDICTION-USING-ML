import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("AccidentsBig.csv")

# Handle missing values
for col in ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Select relevant features
features = ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions', 'longitude', 'latitude']
X = df[features].copy()  # ✅ Create a copy to avoid modifying original data
y = df['Accident_Severity']

# **Manually create label mappings**
weather_mapping = {val: idx for idx, val in enumerate(sorted(df['Weather_Conditions'].unique()))}
road_mapping = {val: idx for idx, val in enumerate(sorted(df['Road_Surface_Conditions'].unique()))}
light_mapping = {val: idx for idx, val in enumerate(sorted(df['Light_Conditions'].unique()))}

# **Apply mapping using `.loc[]` to avoid SettingWithCopyWarning**
X.loc[:, 'Weather_Conditions'] = X['Weather_Conditions'].map(weather_mapping)
X.loc[:, 'Road_Surface_Conditions'] = X['Road_Surface_Conditions'].map(road_mapping)
X.loc[:, 'Light_Conditions'] = X['Light_Conditions'].map(light_mapping)

# Scale numerical features
scaler = StandardScaler()
X.loc[:, ['longitude', 'latitude']] = scaler.fit_transform(X[['longitude', 'latitude']])

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the model, mappings, and scaler
joblib.dump((rf_model, weather_mapping, road_mapping, light_mapping, scaler), "accident_severity_model.pkl")
print("✅ Model trained and saved successfully!")
