import nltk
import tensorflow
import random
import json

import numpy as np
import pickle


from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import load_model
stemmer = LancasterStemmer()




kelimeler = pickle.load(open('kelimeler.pkl', 'rb'))
sınıflar = pickle.load(open('sınıflar.pkl','rb'))
net = load_model('deneme.h5')

def kelime_haznesi(s,kelimeler):
  çanta = [0 for _ in range(len(kelimeler))]
  sentez_kelime = nltk.word_tokenize(s)
  sentez_kelime = [stemmer.stem(klm.lower()) for klm in sentez_kelime]
  for w in sentez_kelime:
    for i, klm in enumerate(kelimeler):
      if klm == w:
        çanta[i] = 1
  return np.array(çanta,dtype=int)

def text(girdi):
  mesaj = input(girdi) 
  snç = net.predict(np.array([kelime_haznesi(mesaj,kelimeler)]))[0]
  sonuç_indisi = np.argmax(snç)
  tag = sınıflar[sonuç_indisi]   
  
  #text = print(text)    
              
text("")   


    









