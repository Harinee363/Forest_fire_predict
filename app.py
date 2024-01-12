from flask import Flask,render_template,Response,url_for,request
import pickle
import pandas as pd 
import numpy as np
app=Flask(__name__)
scaler=pickle.load(open('scaler_model.pkl','rb'))
model=pickle.load(open('predict_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    input_data=scaler.transform(np.array(data).reshape(1,-1))
    print(input_data)
    output=model.predict_proba(input_data)
    print(output)
    output1='{0:.{1}f}'.format(output[0][1], 2)

    if output1>str(0.5):
        return render_template('home.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output1),Hello="kuch karna hain iska ab?")
    else:
        return render_template('home.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output1),Hello="Your Forest is Safe for now")



if __name__=='__main__':
    app.run(debug=True)