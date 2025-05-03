#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('accidents_india.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Fill specific columns if needed (safe fallback, though dropna already used)
df['Sex_Of_Driver'] = df['Sex_Of_Driver'].fillna(df['Sex_Of_Driver'].mean())
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mean())
df['Speed_limit'] = df['Speed_limit'].fillna(df['Speed_limit'].mean())
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mean())
df['Number_of_Pasengers'] = df['Number_of_Pasengers'].fillna(df['Speed_limit'].mean())

# Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="RdYlBu", annot=True, fmt=".1f")
plt.title("Correlation Heatmap")
plt.show()

# Label Encoding
df['Day'] = LabelEncoder().fit_transform(df['Day_of_Week'])
df.drop('Day_of_Week', axis=1, inplace=True)

df['Light'] = LabelEncoder().fit_transform(df['Light_Conditions'])
df.drop('Light_Conditions', axis=1, inplace=True)

df['Severity'] = LabelEncoder().fit_transform(df['Accident_Severity'])
df.drop('Accident_Severity', axis=1, inplace=True)

# Define features and target
x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
r_forest = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=42)
r_forest.fit(x_train, y_train)

# Evaluate model
accuracy = r_forest.score(x_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Predict and show confusion matrix
y_pred = r_forest.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
group_names = ['True Neg','False Pos', 'False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature importance plot
def plot_feature_importances(model):
    plt.barh(range(len(x.columns)), model.feature_importances_, align='center')
    plt.yticks(np.arange(len(x.columns)), x.columns)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.show()

plot_feature_importances(r_forest)

# Save model
pickle.dump(r_forest, open('random_forest_model.pkl', 'wb'))

# Load model and predict on new input
model = pickle.load(open('random_forest_model.pkl', 'rb'))
inputt = [int(x) for x in "2 10 201 10 10 8 3".split(' ')]  # Example input
final = [np.array(inputt)]
prediction = model.predict(final)
print("Prediction for input:", prediction)
