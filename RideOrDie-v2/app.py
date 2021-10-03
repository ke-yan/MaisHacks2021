from flask import Flask
from flask import request, redirect, render_template
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""
Code inspire from: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
Database taken from: https://www.kaggle.com/andrewmvd/divorce-prediction
Other sources used: https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
"""

columnNames = ['Q1','Q2','Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q46', 'Q47', 'Q48', 'Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Divorce']


data = pd.read_csv("divorce_data2.csv", header=None, names=columnNames)
data.head()

#split
dependentVars = ['Q2','Q5','Q7', 'Q8', 'Q9', 'Q12', 'Q14', 'Q19', 'Q23', 'Q24', 'Q27', 'Q28', 'Q29', 'Q30', 'Q33', 'Q37', 'Q39', 'Q42', 'Q45', 'Q47', 'Q53']


# shuffling
shuffledVars = pd.DataFrame(data)
shuffledVars = shuffledVars.sample(frac = 1)
x = shuffledVars[dependentVars]
y = shuffledVars.Divorce

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
model = DecisionTreeClassifier()
model = model.fit(x_train,y_train)
y_pred = model.predict(x_test)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods = ['POST'])
def submit():
    questions = list()
    for x in range (1,22):
        val = int(request.form["Q{}".format(x)])
        questions.append(val)
    prediction = model.predict([questions])[0]
    if prediction == 1 :
         return render_template("saddivorce.html")
    else : 
        return render_template("happymarriage.html")
