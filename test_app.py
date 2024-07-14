import requests
import json

# URL of your Flask application running locally
url = 'http://localhost:5000/predict'

# Example JSON data to send as a POST request
data = [
    {
        "feature1": 0.2,
        "feature2": 0.5,
        "feature3": 0.8,
        # Add more features as required by your model
    },
    {
        "feature1": 0.1,
        "feature2": 0.3,
        "feature3": 0.6,
        # Add more features as required by your model
    }
]

# Convert Python dictionary to JSON string
json_data = json.dumps(data)

# Set headers for the POST request
headers = {'Content-Type': 'application/json'}

# Send POST request to Flask application
response = requests.post(url, data=json_data, headers=headers)

# Print the response JSON
print(response.json())
