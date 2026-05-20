from flask import Flask, render_template, request
import joblib
import numpy as np
import json

app = Flask(__name__)

model = joblib.load("model/house_model.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/compare")
def compare():
    return render_template("compare.html")


@app.route("/trends")
def trends():
    return render_template("trends.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result", methods=["POST"])
def result():

    area = float(request.form["area"])
    bedrooms = float(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    age = float(request.form["age"])

    location = request.form["location"]
    house_type = request.form["type"]

    location_map = {
        "Mumbai":3,
        "Bangalore":2,
        "Delhi":2.5,
        "Hyderabad":2
    }

    type_map = {
        "Apartment":1,
        "Villa":2,
        "Bungalow":3,
        "Penthouse":4
    }

    location_value = location_map.get(location,1)
    type_value = type_map.get(house_type,1)

    features = np.array([[area, bedrooms, bathrooms, age]])

    prediction = model.predict(features)

    price = prediction[0] * location_value * (type_value/2)
    price = round(price,2)


    if price < 4000000:
        category = "Budget Property"
    elif price < 8000000:
        category = "Mid Range Property"
    else:
        category = "Luxury Property"


    confidence = np.random.randint(80,95)

    score = round((location_value*2 + bedrooms + bathrooms)/2,1)

    future_price = round(price * 1.25,2)


    if score > 6:
        advice = "Strong long-term investment potential."
    else:
        advice = "Moderate investment opportunity."


    data = {
        "price":price,
        "score":score
    }

    with open("history.json","a") as f:
        json.dump(data,f)


    return render_template(
        "result.html",
        price=price,
        category=category,
        confidence=confidence,
        score=score,
        forecast=future_price,
        advice=advice,
        house_type=house_type
    )


@app.route("/chat", methods=["POST"])
def chat():

    message = request.form["message"]

    response = "Real estate markets suggest holding property long term increases value."

    return response

@app.route("/compare_result", methods=["POST"])
def compare_result():

    # Property A
    area1 = float(request.form["area1"])
    bed1 = float(request.form["bed1"])
    bath1 = float(request.form["bath1"])
    age1 = float(request.form["age1"])

    # Property B
    area2 = float(request.form["area2"])
    bed2 = float(request.form["bed2"])
    bath2 = float(request.form["bath2"])
    age2 = float(request.form["age2"])

    features1 = np.array([[area1, bed1, bath1, age1]])
    features2 = np.array([[area2, bed2, bath2, age2]])

    price1 = model.predict(features1)[0]
    price2 = model.predict(features2)[0]

    return render_template(
        "compare.html",
        price1=round(price1,2),
        price2=round(price2,2)
    )

@app.route("/history")
def history():

    try:
        with open("history.json") as f:
            data = json.load(f)
    except:
        data = []

    return render_template("history.html", history=data)

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
