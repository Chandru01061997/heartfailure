from flask import Flask, app, render_template, request
from logging import debug
from flask.templating import render_template
import pandas as pd
import joblib

app  = Flask(__name__)
model = joblib.load('heart_failure_detect_model.pkl')

@app.route("/")
def form():
    return render_template('heart.html')

@app.route("/heartfailuredetection", methods=['POST'])
def heartfailuredetection():

    age = request.form.get('age')
    anaemia = request.form.get('anaemia')
    CPK = request.form.get('creatinine_phosphokinase')
    diabetes = request.form.get('diabetes')
    ejection_fraction = request.form.get('ejection_fraction')
    platelets = request.form.get('platelets')
    serum_creatinine = request.form.get('serum_creatinine')
    serum_sodium = request.form.get('serum_sodium')
    sex = request.form.get('sex')
    smoking = request.form.get('smoking')
    time = request.form.get('time')

    dic = {'anaemia':{'Yes':1, 'No':0}, 'diabetes':{'Yes':1, 'No':0}, 'sex':{'Male':1, 'Female':0}, 'smoking':{'Yes':1, 'No':0}}

    anaemia = dic['anaemia'][anaemia]
    diabetes = dic['diabetes'][diabetes]
    sex = dic['sex'][sex]
    smoking = dic['smoking'][smoking]

    prediction = model.predict([[int(age) , int(anaemia) , int(CPK), int(diabetes), int(ejection_fraction), int(platelets), int(serum_creatinine), int(serum_sodium), int(sex), int(smoking), int(time)]])

    output = round(prediction[0] , 2)
    if output == 0:
        predicted_result = "has chance of surviving"
    else:
        predicted_result = "has a survival rate very low"

    return render_template('heart.html' , final_output = f"The person {predicted_result}")

if __name__ == "__main__":
    app.run(debug=True)