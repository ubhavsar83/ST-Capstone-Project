import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from flask import Flask, request, render_template_string

# Load the dataset
data_path = 's_bux.csv'
s_bux = pd.read_csv(data_path)
s_bux['datetime'] = pd.to_datetime(s_bux['datetime'])

# Data converting
s_bux['day_of_week'] = s_bux['datetime'].dt.dayofweek
s_bux['month'] = s_bux['datetime'].dt.month
s_bux['year'] = s_bux['datetime'].dt.year

# Handling outliers
Q1, Q3 = s_bux['close'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
filtered_s_bux = s_bux[(s_bux['close'] >= lower_bound) & (s_bux['close'] <= upper_bound)]

# Data splitting for features and target
X = filtered_s_bux[['open', 'high', 'low', 'volume', 'day_of_week', 'month', 'year']]
y = filtered_s_bux['close']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the model to disk
model_path = 'linear_regression_model.joblib'
dump(model, model_path)

# Load the model
model = load(model_path)

# Initialize the Flask application
app = Flask(__name__)

# HTML template for entering stock data
html_template = """
<!DOCTYPE html>
<html>
<head>
<title>Stock Price Prediction</title>
</head>
<body>
    <h2>Starbucks Stock Closing Price Prediction</h2>
    <form method="post" action="/predict">
        <label for="open">Open Price:</label><br>
        <input type="text" id="open" name="open"><br>
        <label for="high">High Price:</label><br>
        <input type="text" id="high" name="high"><br>
        <label for="low">Low Price:</label><br>
        <input type="text" id="low" name="low"><br>
        <label for="volume">Volume:</label><br>
        <input type="text" id="volume" name="volume"><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Predicted Closing Price: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET'])
def home():
    # Render the HTML form
    return render_template_string(html_template)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form and convert to float
    try:
        open_price = float(request.form['open'])
        high_price = float(request.form['high'])
        low_price = float(request.form['low'])
        volume = float(request.form['volume'])
    except ValueError:
        return "Please enter valid numbers for all input fields."

    # Prepare features array for prediction
    features = [[open_price, high_price, low_price, volume,
                 s_bux['datetime'].dt.dayofweek.iloc[0],  # Assuming today's date for simplicity
                 s_bux['datetime'].dt.month.iloc[0],
                 s_bux['datetime'].dt.year.iloc[0]]]

    # Use the model to predict the closing price
    prediction = model.predict(features)

    # Render the HTML form with the prediction result
    return render_template_string(html_template, prediction=round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
