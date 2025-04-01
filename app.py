import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Define features and target
X = crop.drop(columns=['label'])
y = crop['label']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Normalize feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train optimized model
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Flask app setup
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chatboot')
def chatboot():
    return render_template("chatboot.html")

@app.route('/weather')
def weather():
    return render_template("weather.html")

@app.route('/market')
def market():
    return render_template("market.html")

@app.route('/crop')
def crop():
    return render_template("crop.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Fetch input values from the form
        input_features = [
            float(request.form.get('Nitrogen', 0)),
            float(request.form.get('Phosphorus', 0)),
            float(request.form.get('Potassium', 0)),
            float(request.form.get('Temperature', 0)),
            float(request.form.get('Humidity', 0)),
            float(request.form.get('Ph', 0)),
            float(request.form.get('Rainfall', 0))
        ]

        # Prepare input features
        feature_array = np.array(input_features).reshape(1, -1)

        # Apply MinMaxScaler
        scaled_features = scaler.transform(feature_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        crop_name = le.inverse_transform(prediction)[0]
        result = f"{crop_name} is the best crop to be cultivated right now."

        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

@app.route("/api/predict", methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        input_features = [
            float(data.get('Nitrogen', 0)),
            float(data.get('Phosphorus', 0)),
            float(data.get('Potassium', 0)),
            float(data.get('Temperature', 0)),
            float(data.get('Humidity', 0)),
            float(data.get('Ph', 0)),
            float(data.get('Rainfall', 0))
        ]

        # Prepare input features
        feature_array = np.array(input_features).reshape(1, -1)

        # Apply MinMaxScaler
        scaled_features = scaler.transform(feature_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        crop_name = le.inverse_transform(prediction)[0]

        return jsonify({"crop_recommendation": crop_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
