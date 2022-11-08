import nltk
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

sözler = json.loads(open('metin.json').read())
sınıflar = []
kelimeler = []
dosyalar = []
red_kelimeler=['!','?','.',',']


for niyet in sözler['intents']:
    for şablon in niyet['patterns']:
        klm = nltk.word_tokenize(şablon)
        kelimeler.extend(klm)
        dosyalar.append((klm, niyet['tag']))
        if niyet['tag'] not in sınıflar:
          sınıflar.append(niyet['tag'])
kelimeler = [stemmer.stem(word) for word in kelimeler if word not in red_kelimeler]
kelimeler = sorted(set(kelimeler))
sınıflar = sorted(sınıflar)
pickle.dump(kelimeler, open('kelimeler.pkl','wb'))
pickle.dump(sınıflar, open('sınıflar.pkl','wb'))  

egitim = []
çıktı = [] 
boş_çıktı = [0 for _ in range(len(sınıflar))]

for dosya in dosyalar:  
    çanta = []
    kelime_katmanları = dosya[0]
    kelime_katmanları = [stemmer.stem(word.lower()) for word in kelime_katmanları]
    for word in kelimeler:
      çanta.append(1) if word in kelimeler else çanta.append(0)
    çıkış_satırı = boş_çıktı[:]
    çıkış_satırı[sınıflar.index(dosya[1])] = 1
    egitim.append([çanta,boş_çıktı])
    çıktı.append(çıkış_satırı)
random.shuffle(egitim)     
egitim = np.array(egitim,dtype=object) 
çıktı = np.array(çıktı,dtype=object)
egi_x = list(egitim[:,0])
egi_y = list(egitim[:,1])
net=Sequential()
net.add(Dense(128, input_shape=(len(egi_x[0]),), activation='relu'))
net.add(Dropout(0.5))
net.add(Dense(64,activation='relu'))
net.add(Dropout(0.5))
net.add(Dense(len(egi_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum = 0.9, nesterov=True)
net.compile(loss='categorical_crossentropy',optimizer = sgd, metrics=['accuracy'])
egitilenveri=net.fit(np.array(egi_x,dtype=int),np.array(egi_y,dtype=int), epochs=1000, batch_size=5, verbose=1)
net.save('deneme.h5', egitilenveri)
print("Tamam!")




