# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:00:39 2020

@author: Diwakar
"""

from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

#Open a file,where you want to store the data
file=open('model.pk','rb')
clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDic=request.form
        fever=int(myDic['fever'])
        age=int(myDic['age'])
        bodypain=int(myDic['bodypain'])
        runnynose=int(myDic['runnynose'])
        diffbreath=int(myDic['diffbreath'])
        inputFeatures=[fever,bodypain,age,runnynose,diffbreath]
        infProb=clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!   '+  str(infProb)
if __name__=="__main__":
    app.run(debug=True)
    