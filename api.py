from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('E:/ai model/ai_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get input data from request
    inputs = np.array(data['inputs']).reshape(1, -1)  # Convert to numpy array
    prediction = model.predict(inputs)  # Get prediction
    return jsonify({'prediction': prediction.tolist()})  # Return prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)
