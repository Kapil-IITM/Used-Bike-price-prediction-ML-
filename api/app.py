from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

# ✅ FIX 1: correct template & static path
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ✅ FIX 2: correct model path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Brand list
brands = ['Bajaj', 'Royal Enfield', 'Hero', 'Honda', 'Yamaha', 'TVS', 'KTM', 'Suzuki', 'other']


@app.route('/')
def index():
    return render_template('index.html', brands=brands)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form.get('brand')
        kms = float(request.form.get('kms_driven'))
        age = float(request.form.get('age'))
        power = float(request.form.get('power'))
        owner = request.form.get('owner')

        owner_map = {
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3,
            'Fourth Owner Or More': 4
        }
        owner = owner_map.get(owner, 1)

        kms_log = np.log1p(kms)
        km_per_year = kms_log / (age + 1)

        input_df = pd.DataFrame([{
            'brand': brand,
            'kms_driven': kms_log,
            'age': age,
            'power': power,
            'owner': owner,
            'km_per_year': km_per_year
        }])

        pred = model.predict(input_df)
        final_price = round(np.expm1(pred[0]), 2)

        return render_template('index.html', prediction=final_price, brands=brands)

    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == '__main__':
    app.run(debug=True)