from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('phishing_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        if isinstance(data, list):
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data)
        else:
            # Convert a single dictionary to a DataFrame
            df = pd.DataFrame([data])
        
        # Ensure the DataFrame columns match the model's expected features
        df = df[model.feature_names_in_]

        # Scale the features
        features_scaled = scaler.transform(df)

        # Make predictions
        predictions = model.predict(features_scaled)

        return jsonify({'predictions': predictions.tolist()})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Placeholder for retraining logic
        return jsonify({'message': 'Model retrained successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
