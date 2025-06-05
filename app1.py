from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import webbrowser
import threading

app = Flask(__name__)

# Sample data
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2, 2.2, 2.9, 3, 3.2, 3.2, 3.7, 
                        3.9, 4, 4, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6,
                        6.8, 7.1, 7.9, 8.2, 8.7, 9, 9.5, 9.6, 10.3, 10.5],
    'PreviousSalary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
                       63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
                       91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872],
    'Salary': [42000, 48500, 40000, 47000, 43000, 61000, 65000, 58000, 71000, 62000,
               70000, 61000, 63000, 63500, 67000, 74000, 73000, 89000, 87000, 100000,
               99000, 106000, 108000, 122000, 117000, 113000, 127000, 123000, 132000, 133000]
}

# Prepare the model
df = pd.DataFrame(data)
X = df[['YearsExperience', 'PreviousSalary']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        years_exp = float(request.form['YearsExperience'])
        prev_salary = float(request.form['PreviousSalary'])
        input_features = np.array([[years_exp, prev_salary]])
        prediction = model.predict(input_features)
        predicted_salary = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted Salary: ₹ {predicted_salary}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Auto-launch in browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
