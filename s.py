import pandas as pd

# Load dataset
df = pd.read_csv("AccidentsBig.csv")

# Print unique values for each categorical column
print("Weather Conditions:", df["Weather_Conditions"].unique())
print("Road Surface Conditions:", df["Road_Surface_Conditions"].unique())
print("Light Conditions:", df["Light_Conditions"].unique())
#E:\project\Road Accident Severity Prediction>python s.py
#Weather Conditions: [2 1 9 3 8 4 5 7 6]
#Road Surface Conditions: [ 2  1  4  3  5 -1]
#Light Conditions: [1 4 7 5 6]