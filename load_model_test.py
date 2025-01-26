from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model('E:/ai model/ai_model.h5')
print("Model loaded successfully!")

# Accept custom input from the user
print("Enter 10 numbers separated by spaces (e.g., 0.1 0.2 0.3 ...):")
user_input = input()  # Read user input as a string
test_data = np.array([float(x) for x in user_input.split()]).reshape(1, -1)  # Convert input to numpy array

# Make a prediction
prediction = model.predict(test_data)
print("Model prediction:", prediction)
