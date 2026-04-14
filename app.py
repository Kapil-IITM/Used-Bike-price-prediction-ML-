from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model (pipeline)
model = pickle.load(open('model.pkl', 'rb'))

# Brand list (same as training)
brands = ['Bajaj', 'Royal Enfield', 'Hero', 'Honda', 'Yamaha', 'TVS', 'KTM', 'Suzuki', 'other']

@app.route('/')
def index():
    return render_template('index.html', brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔹 Get input from form
        brand = request.form.get('brand')
        kms = float(request.form.get('kms_driven'))
        age = float(request.form.get('age'))
        power = float(request.form.get('power'))
        owner = request.form.get('owner')

        # 🔹 Owner mapping
        owner_map = {
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3,
            'Fourth Owner Or More': 4
        }
        owner = owner_map.get(owner, 1)

        # 🔹 Apply SAME transformations as training
        kms_log = np.log1p(kms)
        km_per_year = kms_log / (age + 1)

        # 🔹 Create dataframe (VERY IMPORTANT: same column names)
        input_df = pd.DataFrame([{
            'brand': brand,
            'kms_driven': kms_log,
            'age': age,
            'power': power,
            'owner': owner,
            'km_per_year': km_per_year
        }])

        # 🔹 Prediction
        pred = model.predict(input_df)

        # 🔹 Convert back from log
        final_price = round(np.expm1(pred[0]), 2)

        return render_template('index.html', prediction=final_price, brands=brands)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)