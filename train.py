# train.py
import pandas as pd
from neural_network import build_model

# Load and create data inside train.py ONLY
data = pd.read_csv('creditcard.csv')
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# Initialize the model structure
model = build_model()

# Train and save
model.fit(X, y, epochs=20, batch_size=2048, verbose=1)
model.save('fraud_detection_model.keras')
print("Model saved successfully!")
