import requests

# API endpoint
url = 'http://127.0.0.1:5000/predict'

# Input data (replace with your test input)
data = {
    "inputs": [0.5, 0.2, 0.1, 0.6, 0.7, 0.4, 0.3, 0.8, 0.9, 0.2]
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print("Response from API:", response.json())
