import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Path to the dataset
dataset_path = 'dataset_full.csv'

# Load the dataset
df = pd.read_csv(dataset_path)

# Actual name of your target variable
target_variable = 'phishing'

# Split the dataset into features and target variable
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

print("Model training completed and saved as 'model.pkl'")
