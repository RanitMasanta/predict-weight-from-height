# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

X = df[['Height']]
y = df['Weight']

model = LinearRegression()
model.fit(X, y)

# Save the model
with open('weight_model.pkl', 'wb') as f:
    pickle.dump(model, f)
