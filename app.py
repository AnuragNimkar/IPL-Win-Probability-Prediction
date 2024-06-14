from flask import Flask, render_template, request, send_file
import io
import base64
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# data = pd.DataFrame({'Models': ['LR', 'SVM', 'KNN', 'DT', 'RF', 'GB'], 'ACC': [78.688525, 80.327869, 83.770492, 70.491803, 85.245902, 80.327869]})
# acc_array = data.values


warnings.filterwarnings("ignore")

model = pickle.load(
    open('C:\\Users\\Lenovo\\Desktop\\DSBDA MINI PROJECT\\pipe.pkl' , 'rb'))

app = Flask(__name__)



@app.route('/')
def landing():
    return render_template('landing.html',  methods=['GET', 'POST'])


@app.route('/home', methods=['GET', 'POST'])


    
def home():
    r = 0
    if request.method == 'POST':
        input_text = []
        for i in range(1, 10):
            it = 'input_text' + str(i)
            print("requst : ",it," : ",request.form[it])
            input_text.append(int(request.form[it]))


        my_dataframe = pd.DataFrame({
            'battingteam': input_text[0],
            'bowlingteam':  input_text[1],
            'host_city':  input_text[2],
            'runs_left':  input_text[3],
            'balls_left': input_text[4],
            'wickets':  input_text[5],
            'total_runs_x':  input_text[6],
            'crr':  input_text[7],
            'rrr':  input_text[8],
            
        }, index=[0])

        result = model.predict_proba(my_dataframe)
        loss = result[0][0]
        win = result[0][1]
        
        print("result is : ",result[0])

        print(result)
        
        r = result[0]
       

        

    

        return render_template('home.html', output=r)

    return render_template('home.html')







if __name__ == '__main__':
    app.run()
