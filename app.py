# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
with open('weight_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    height = float(request.form['height'])
    weight_pred = model.predict([[height]])[0]
    return render_template('index.html', prediction=round(weight_pred, 2))

if __name__ == '__main__':
    app.run(debug=True)
