import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
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

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/chatboot')
def chatboot():
    return render_template("chatboot.html")
 
    
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Fetching input values from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Prepare input features
        feature_list = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)

        # Apply MinMaxScaler
        scaled_features = scaler.transform(feature_list)

        # Make prediction
        prediction = model.predict(scaled_features)
        crop_name = le.inverse_transform(prediction)[0]
        result = f"{crop_name} is the best crop to be cultivated right now."

        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
