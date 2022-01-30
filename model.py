#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re


# In[40]:


data = pd.read_csv('feedback_dataset.csv')


# In[41]:


data.info()


# In[43]:


def to_sentiment(sentiment):
  sentiment = sentiment
  if sentiment == 0:
    return 'negative'
  else: 
    return 'positive'
data['sentiment'] = data.sentiment.apply(to_sentiment)


# In[44]:


def preProcess_data(text):
   text = text.lower()
   new_text = re.sub('[^a-zA-z0-9\s]','',text)
   new_text = re.sub('rt', '', new_text)
   return new_text

data['text'] = data['text'].apply(preProcess_data)


# In[51]:


max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, 28) 

Y = pd.get_dummies(data['sentiment']).values


# In[52]:


Y


# In[53]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20)


# In[56]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128,recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


# In[57]:


batch_size = 512

model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, validation_data=(X_test, Y_test))


# In[58]:


# we have saved our model in ‘hdf5’ format
model.save('sentiment.h5')


# In[ ]:




