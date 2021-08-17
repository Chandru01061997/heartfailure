from flask import Flask, app, render_template, request
from logging import debug
from flask.templating import render_template
import pandas as pd
import joblib

app  = Flask(__name__)
model = joblib.load('finalized_model.pkl')

@app.route("/")
def form():
    return render_template('heart.html')

@app.route("/heartfailuredetection", methods=['POST'])
def heartfailuredetection():

    age = request.form.get('age')
    anaemia = request.form.get('anaemia')
    diabetes = request.form.get('diabetes')
    high_blood_pressure = request.form.get('high_blood_pressure')
    platelets = request.form.get('platelets')
    serum_sodium = request.form.get('serum_sodium')
    time = request.form.get('time')
    smoking = request.form.get('smoking')

    dic = {'anaemia':{'Yes':1, 'yes':1, 'No':0, 'no':0}, 'diabetes':{'Yes':1, 'yes':1, 'No':0, 'no':0}, 
    'HBP':{'Yes':1, 'yes':1, 'No':0, 'no':0}, 'smoking':{'Yes':1, 'yes':1, 'No':0, 'no':0}}

    anaemia = dic['anaemia'][anaemia]
    diabetes = dic['diabetes'][diabetes]
    high_blood_pressure = dic['HBP'][high_blood_pressure]
    smoking = dic['smoking'][smoking]

    prediction = model.predict([[int(age) , int(anaemia) , int(diabetes), int(high_blood_pressure), int(platelets), int(serum_sodium), int(time), int(smoking)]])

    output = round(prediction[0] , 2)
    if output == 0:
        predicted_result = "Survive"
    else:
        predicted_result = "Not Survive"

    return render_template('heart.html' , final_output = f"The person will {predicted_result}")

if __name__ == "__main__":
    app.run(debug=True)
