from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    salary = int(request.form['salary'])

    data = np.array([[gender, age, salary]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Customer WILL purchase"
    else:
        result = "Customer will NOT purchase"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)