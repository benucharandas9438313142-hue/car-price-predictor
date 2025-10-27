from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained Random Forest model
with open("car.pkl", "rb") as f:
    model = pickle.load(f)

# 📘 Load dataset for dropdowns
df = pd.read_csv("car.csv")

# Sort unique dropdown values
brands = sorted(df["brand"].dropna().unique().tolist())
models = sorted(df["model"].dropna().unique().tolist())
car_names = sorted(df["car_name"].dropna().unique().tolist())

# 🏠 Home route
@app.route("/")
def home():
    return render_template(
        "index.html",
        brands=brands,
        models=models,
        car_names=car_names,
        predicted_price=""
    )

# 🔁 Route to get models based on selected brand
@app.route("/get_models/<brand>")
def get_models(brand):
    filtered_models = sorted(df[df["brand"] == brand]["model"].dropna().unique().tolist())
    return jsonify(filtered_models)

# 🔁 Route to get car names based on selected brand and model
@app.route("/get_cars/<brand>/<model>")
def get_cars(brand, model):
    filtered_cars = sorted(
        df[(df["brand"] == brand) & (df["model"] == model)]["car_name"].dropna().unique().tolist()
    )
    return jsonify(filtered_cars)

# 🔮 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        brand = request.form['bn']
        model_name = request.form['cm']
        car_name = request.form['cn']
        vehicle_age = float(request.form['va'])
        km_driven = float(request.form['kd'])
        seller_type = request.form['st']
        fuel_type = request.form['ft']
        transmission_type = request.form['tt']
        mileage = float(request.form['mg'])
        engine = float(request.form['eg'])
        max_power = float(request.form['mp'])
        seats = int(request.form['seat'])

        # Combine all input features into one list
        input_data = [
            brand, model_name, car_name,
            vehicle_age, km_driven, seller_type,
            fuel_type, transmission_type, mileage,
            engine, max_power, seats
        ]

        # ⚙️ Encode categorical data (temporary hash encoding)
        encoded_features = []
        for val in input_data:
            if isinstance(val, str):
                encoded_features.append(hash(val) % 1000)
            else:
                encoded_features.append(val)

        # Convert to numpy array
        final_features = np.array(encoded_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)[0]

        # Display result
        return render_template(
            "index.html",
            brands=brands,
            models=models,
            car_names=car_names,
            predicted_price=f"Predicted Car Price: ₹{round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            brands=brands,
            models=models,
            car_names=car_names,
            predicted_price=f"⚠️ Error: {str(e)}"
        )

# 🚀 Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
