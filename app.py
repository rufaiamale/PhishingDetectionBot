from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

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
    # Bind to PORT if defined, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
