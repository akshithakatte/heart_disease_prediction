import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Route for the root URL (to render the index.html page)
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction functionality (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the request
        data = request.get_json()

        # Extract the values from the data
        sex = data.get('sex')
        age = data.get('age')
        cp = data.get('cp')
        trestbps = data.get('trestbps')
        chol = data.get('chol')
        fbs = data.get('fbs')
        restecg = data.get('restecg')
        thalach = data.get('thalach')
        exang = data.get('exang')
        oldpeak = data.get('oldpeak')
        slope = data.get('slope')
        ca = data.get('ca')
        thal = data.get('thal')

        # Create a feature vector (based on your model's requirements)
        feature_vector = [[sex, age, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

        # Predict using the model
        prediction = model.predict(feature_vector)[0]

        # Respond with the prediction result
        result = {
            'prediction': prediction
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
