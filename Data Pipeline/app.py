import numpy as np
from fastapi import FastAPI, Request, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = directory='templates'

data = pd.read_csv('feedback_dataset.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

def preProcess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

def my_pipeline(text):
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X

@app.get('/index', response_class=HTMLResponse)  # route to display the home page
def homePage():
    return templates.TemplateResponse("templates/index.html",{
        'request': request
    })

@app.post('/prediction')
def index(text:str = Form(...)):
    clean_text = my_pipeline(text) #clean, and preprocess the text through pipeline
    loaded_model = tf.keras.models.load_model('sentiment.h5') #load the saved model 
    predictions = loaded_model.predict(clean_text) #predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    if sentiment==0:
        t_sentiment = 'negative' #set appropriate sentiment
    elif sentiment==1:
        t_sentiment = 'postive'
    return { #return the dictionary for endpoint
        "ACTUALL SENTENCE": text,
        "PREDICTED SENTIMENT": t_sentiment,
        "Probability": probability}
    return templates.TemplateResponse('templates/prediction.html', prediction=prediction)

