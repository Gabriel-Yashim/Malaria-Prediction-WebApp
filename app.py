# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:24:33 2022

@author: YASHIM GABRIEL
"""

import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('GradientBoost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():   
     sex = int(request.form['SEX'])
     age = int(request.form['AGE'])
     semester = int(request.form['SEMESTER'])
     Temperature = float(request.form['BODY TEMPERATURE'])
     bp = int(request.form['BP'])
     cata = int(request.form['CATARRH'])
     cough = int(request.form['COUGH'])
     h_Ache = int(request.form['HEADACHE'])
     fever = int(request.form['FEVER'])
     cold = int(request.form['COLD'])
     b_Pain = int(request.form['BODY PAIN'])
     j_Pain = int(request.form['JOINT PAIN'])
     dizz = int(request.form['DIZZINESS'])
     b_Weak = int(request.form['BODY WEAKNESS'])
     dia = int(request.form['DIARRHOEA'])
     dy_Stool = int(request.form['DYSENTERY STOOL'])
     ab_Pain = int(request.form['ABDOMINAL PAIN'])
     swt = int(request.form['SWEATING'])
     slvy = int(request.form['SALIVATORY'])
     s_Throat = int(request.form['SORE THROAT'])
     vom = int(request.form['VOMITTING'])
     int_Heat = int(request.form['INTERNAL HEAT'])
     m_Bit = int(request.form['MOUTH BITTERNESS'])
     sleep = int(request.form['SLEEPLESS'])
     tired = int(request.form['TIREDNESS'])
     no_App = int(request.form['NO APETITE'])
     b_Diff = int(request.form['BREATHING DIFFICULTY'])
     ch_Pain = int(request.form['CHEST PAIN'])
     nausea = float(request.form['NAUSEA'])
     eye_Ache = int(request.form['EYE ACHE'])
     r_Eye = int(request.form['RED EYE'])
    
    
     final_features = np.array([[sex,age,semester,Temperature,bp,cata,cough,h_Ache,
                                 fever,cold,b_Pain,j_Pain,dizz,b_Weak,dia,dy_Stool,
                                 ab_Pain,swt,slvy,s_Throat,vom,int_Heat,m_Bit,sleep,
                                 tired,no_App,b_Diff,ch_Pain,nausea,eye_Ache,r_Eye]])
     prediction = model.predict(final_features)
    
     output = prediction[0]
    
     return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)