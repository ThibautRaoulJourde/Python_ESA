# Projet Python - Sujet B - BUVAT Antoine / RAOUL JOURDE Thibaut / YE Maxime

#Prérequis : 

#Installation des packages

pip install pandas
pip install keras
pip install tensorflow
pip install nltk
pip install matplotlib
pip install Scikit
pip install sklearn
pip install gensim



#Importation des packages utilisés par la suite : 

import pandas as pd
import re
from nltk.tokenize import WordPunctTokenizer as wpt
import json
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import numpy as np
import os
from collections import Counter
import logging
import time
import itertools
import gensim



###########################################################################
###########################################################################
###########################################################################


#1ère Partie : DATA PREPARATION 

# L'objectif dans cette première partie est de réaliser un nettoyage de la base de données 
# Nettoyages effectués sur les tweets : 
# - Suppression des bouts de code HTML
# - Suppression des tags (@utilisateur)
# - Suppression des liens (Https://...)


#Import du fichier csv 

df = pd.read_csv('/Users/util/Desktop/M2 ESA/Python/Projet python/training.1600000.processed.noemoticon.csv', encoding ="ISO-8859-1" , names=["target", "ids", "date", "flag", "user", "text"])

#On renseigne ici le sens des valeurs de la variable "target" (=sentiment du tweet)
# 0 : Tweet négatif / 2 : Tweet Neutre / 4: Tweet Positif

decode_map = {0: "Négatif", 2: "Neutre", 4: "Positif"}

#Fonction pour affecter les sentiments aux entiers de la colonne target

def decode_sentiment(label):
    return decode_map[int(label)]

%%time
df.target = df.target.apply(lambda s: decode_sentiment(s))


#Suppression des stopwords (les mots les + utilisés mais qui 
# n'aident en rien à la compréhension du texte
# 
# Ex: "et" "à" "le" en FR
# 
# Ici les tweets sont en anglais)

stop_words = stopwords.words("english")

#Racinisation des tweets (on cherche à conserver
# uniquement la racine des mots 
# 
# Ex : "Ils sont attentifs"
#  -> Après racinisation : "être" "attentif"
# 
# Le "attentifs" est ramené au singulier et "sont" est ramené au verbe "être")
stemmer = SnowballStemmer("english")



#Fonction pour nettoyer les tweets : 
#Corrections à apporter sur la base de données : 
# - Suppressions des mentions "@toto"
# - Suppression des liens https
# - Suppression des résidus de code HTML


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)



%%time
df.text = df.text.apply(lambda s: preprocess(s))



###########################################################################
###########################################################################
###########################################################################

# 2ème Partie : Mise en place du modèle NLP 


#On crée 2 sous échantillons : 
#Echantillon d'apprentissage (train): 70% des données
#Echantillon de test (test): 30% des données restantes
#Seed= 12345


train, test = train_test_split(df, test_size=0.3, random_state=12345)


#On sépare chaque mot de la colonne "text" 
#Les mots individuels sont stockés dans la liste "mot"
%%time
mot = [_text.split() for _text in train.text] 


#Voir documentation sur Gensim 

gen_model = gensim.models.word2vec.Word2Vec(size=500, window=3, min_count=10, workers=8)

gen_model.build_vocab(mot)


lmot = gen_model.wv.vocab.keys()
nmot = len(lmot)



%%time
gen_model.train(mot, total_examples=len(mot), epochs=1)



#Tokenisation

%%time
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.text)

nmot = len(tokenizer.word_index) + 1

pickle.dump(tokenizer, open("/Users/util/Desktop/M2 ESA/Python/Projet python/tokenizer.pkl", "wb"), protocol=0)


# nmot = nombre total de mots pris en compte pour entrainer le modèle

%%time
x_train = pad_sequences(tokenizer.texts_to_sequences(train.text), maxlen=250)
x_test = pad_sequences(tokenizer.texts_to_sequences(test.text), maxlen=250)

#Ajout du sentiment "Neutre" 

labels = train.target.unique().tolist()
labels.append('Neutre')
labels

encoder = LabelEncoder()
encoder.fit(train.target.tolist())

y_train = encoder.transform(train.target.tolist())
y_test = encoder.transform(test.target.tolist())


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train[:10]

embedding_matrix = np.zeros((nmot, 500))
for word, i in tokenizer.word_index.items():
  if word in gen_model.wv:
    embedding_matrix[i] = gen_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = Embedding(nmot, 500, weights=[embedding_matrix], input_length=250, trainable=False)


model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

#Ajustement du modèle : 

%%time
ajust = model.fit(x_train, y_train,batch_size=500,epochs=1,validation_split=0.1,verbose=1)

%%time
perf = model.evaluate(x_test, y_test, batch_size=500)

#Sauvegarde
model.save("/Users/util/Desktop/M2 ESA/Python/Projet python/model.h5")


#Fonction qui décode le sentiment en fonction du texte 
#Dépendant du score calculé : 
# Score : 
# =< 0.35 : Sentiment négatif associé au tweet
# 0.35 < x < 0.65 : Sentiment neutre associé au tweet
# >= 0.65 : Sentiment positif associé au tweet 

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = "Tweet à sentiment neutre"
        if score <= 0.35:
            label = "Tweet à sentiment négatif"
        elif score >= 0.65:
            label = "Tweet à sentiment positif"

        return label
    else:
        return "Tweet à sentiment négatif" if score < 0.5 else "Tweet à sentiment positif"

#Fonction permettant de prédire le sentiment d'un texte renseigné en paramètre


def predict(text, include_neutral=True):
    
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=250)

    score = model.predict([x_test])[0]
   
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}  


###########################################################################
###########################################################################
###########################################################################

#3ème Partie : Faire appel aux fichiers contenant les modèles (pour ne pas 
# avoir à tout relancer) : 

model = keras.models.load_model('C:/Users/util/Desktop/M2 ESA/Python/Python/model.h5')

with open("C:/Users/util/Desktop/M2 ESA/Python/Python/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


#Fonctions à lancer : 

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = "Tweet à sentiment neutre"
        if score <= 0.35:
            label = "Tweet à sentiment négatif"
        elif score >= 0.65:
            label = "Tweet à sentiment positif"

        return label
    else:
        return "Tweet à sentiment négatif" if score < 0.5 else "Tweet à sentiment positif"

#Fonction permettant de prédire le sentiment d'un texte renseigné en paramètre


def predict(text, include_neutral=True):
    
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=250)

    score = model.predict([x_test])[0]
   
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}  


###########################################################################
###########################################################################
###########################################################################

# 4ème Partie : Utilisation de la fonction : 

#Exemple d'utilisation : 

predict("great")
predict("maybe")
predict("this is bad")
predict(" ")
predict(" ")
predict(" ")
predict(" ")
predict(" ")
predict(" ")



###########################################################################
###########################################################################
###########################################################################

# 5ème Partie : Évaluation des performances de notre modèle :

# On va maintenant utilisé notre fonction sur notre base test:

# On crée un dataframe vide quinous servira ensuite de table de résultats

columns = ['label','score']

#On importe notre base test

df_ingredients_recipes = pd.DataFrame(columns=columns)

#On réalise maintenant une boucle qui appliquera notre fonction pour
#chacun des tweet de notre table

df_results=pd.DataFrame(columns=columns)

df_test = pd.read_csv('C:/Users/util/Desktop/M2 ESA/Python/Projet python/testdata.manual.2009.06.14.csv' , names=["target", "ids", "date", "flag", "user", "text"])

for i in range(0,df_test['text'].nunique()-1,1):
    test=df_test['text']
    test
    test2=predict(test[i])
    test2
    results=pd.DataFrame(test2, index=[i])
    df_results=df_results.append(results)

df_results

#On transforme notre variable target de manière à obtenir le même 
#type de vecteur que celui des prévisions

decode_map_2 = {0: "Tweet à sentiment négatif", 2: "Tweet à sentiment neutre", 4: "Tweet à sentiment positif"}

def decode_sentiment_2(label):
    return decode_map_2[int(label)]

df_results['target'] = df_test.target.apply(lambda s: decode_sentiment_2(s))
df_results

#On calcule le pourcentage de bonne classification

df_results['conf']=df_results["label"]==df_results["target"]
df_results['conf']
df_results.groupby('conf').count()/497

#63.58% des sentiments de notre échantillons ont été analysés
#correctement par notre fonction


    

#Liens utiles pour la compréhension 

#Stopword et Racinisation 

https://openclassrooms.com/fr/courses/4470541-analysez-vos-donnees-textuelles/4854971-nettoyez-et-normalisez-les-donnees

#Package/ Fonction Gensim (pour entrainer les vecteurs de mots)
https://radimrehurek.com/gensim/models/word2vec.html

