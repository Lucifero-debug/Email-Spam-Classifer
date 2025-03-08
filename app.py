from flask import Flask,render_template,request
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app=Flask(__name__)
rfc = pickle.load(open('rfc.pkl','rb'))
tfd=pickle.load(open('tfd.pkl','rb'))

port_Stem=PorterStemmer()
def streming(content):
    stremed_content=re.sub('[^a-zA-Z]',' ',content)
    stremed_content=stremed_content.lower()
    stremed_content=stremed_content.split()
    stremed_content=[port_Stem.stem(word) for word in stremed_content if not word in stopwords.words('english')]
    stremed_content=' '.join(stremed_content)
    return stremed_content

@app.route("/")
def home():
    return render_template('index.html',prediction=None)

@app.route("/submit",methods=['POST','GET'])
def helper():
    if request.method=='POST':
        text=request.form['text']
        print("text",text)
        test=streming(text)
        test=[test]
        test=tfd.transform(test)
        pred=rfc.predict(test)[0]
        if pred==0:
            res='Not Spam'
        else:
            res='Spam'
        return render_template("index.html", prediction=res)
        
if __name__=='__main__':
    app.run(debug=True)