# predictions.py
import numpy as np
from tensorflow.keras.models import load_model

# Load your model from disk
model = load_model('fraud_detection_model.keras')

single_transaction = np.array([[
    -1.35,  0.07,  2.53,  1.37, -0.33,  0.46,  0.23, -0.11,  0.06,  0.12, 
    -0.18,  0.13, -0.02,  0.50, -0.20,  0.10,  0.05, -0.01,  0.02, -0.03, 
     0.04, -0.05,  0.01, -0.02,  0.03, -0.01,  0.02, -0.04,  0.55
]])

# Make predictions
prediction = model.predict(single_transaction)
predicted_label = (prediction > 0.5).astype(int)

print(f"Fraud Probability: {prediction[0][0]:.4f}")
print(f"Predicted Class: {predicted_label[0][0]}")
