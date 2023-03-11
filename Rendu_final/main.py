import uvicorn 
from fastapi import FastAPI
import onnxruntime as rt
import tf2onnx
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from keras_preprocessing.sequence import pad_sequences
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json




#********************************************************Function*************************************************************

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   

from transformers import BertTokenizer
import torch
import onnx
import onnxruntime as ort

class_names = ['positive','negative', 'neutral']
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
ort_sess = ort.InferenceSession('C:\\Users\\Pénichon\\Desktop\\M2_TIDE\\TIDE_S1\\Algo_python\\Projet_NLP\\Rendu_final\\\\modele_sentiment_BERT\\torch-model.onnx')

def infer_onnx_model(review_text__):
    encoded_review = tokenizer.encode_plus(
                              review_text__,
                              max_length=512,
                              add_special_tokens=True,
                              return_token_type_ids=False,
                              pad_to_max_length=True,
                              return_attention_mask=True,
                              return_tensors='pt',
                            )
    outputs = ort_sess.run(None, {'input_ids': encoded_review['input_ids'].cpu().detach().numpy(),
                              'attention_mask':encoded_review['attention_mask'].cpu().detach().numpy()})
    outputs = torch.from_numpy(outputs[0])
    _, prediction = torch.max(outputs, dim=1)
    return class_names[prediction]




#list of point fort and point faible : 
import numpy as np
import re

#NER BERT :   
from simpletransformers.ner import NERModel,NERArgs
from simpletransformers.ner import NERModel

#le path pour charger le modèle
path_BERT_NER ='C:\\Users\\Pénichon\\Desktop\\M2_TIDE\\TIDE_S1\\Algo_python\\Projet_NLP\\Rendu_final\\modele_NER_BERT\\NER_BERT_Model'
model2 = NERModel("bert", path_BERT_NER, use_cuda=False)


#Ce code définit une fonction fort_faible() qui prend en argument une chaîne de caractères text et qui renvoie une liste de 
#dictionnaires contenant des mots et leurs étiquettes de nommage d'entités.
def fort_faible(text : str) :
    text = text.lower() #la chaîne de caractères est convertie en minuscule
    #on séprare en différentes phrases en utilisant une expression régulière
    text = re.split(r'[\.!?;,]| but | whereas | yet | althought | though | still | furthermore | moreover | even though | on the other hand | in spite of ', text)
    #Ensuite, la fonction crée une liste vide appelée list_word qui sera utilisée pour stocker les mots et leurs étiquettes.
    list_word = []
    
    #Pour chaque phrase dans text, la fonction calcule un score de sentiment en utilisant le modèle model et en appelant la 
    #méthode polarity_scores(). Si le score est supérieur à 0,3, la fonction crée un nouveau modèle appelé model2 qui sera 
    #utilisé pour détecter les entités nommées dans la phrase. La fonction utilise alors le modèle model2 pour prédire les entités 
    #nommées dans la phrase et stocke le résultat dans une variable pred.
    for phrase in text :
        score = infer_onnx_model(phrase) 
        
        if score == 'positive' :
            prediction, model_output = model2.predict([phrase])
            pred = prediction
            
            #Enfin, la fonction itère sur chaque mot et étiquette dans pred et ajoute le mot et son étiquette à la liste 
            #list_word s'il n'est pas marqué comme "O", c'est-à-dire s'il n'est pas une entité nommée.
            i=0
            while i < len(pred[0]) :
                word, label = pred[0][i].popitem()
                word_label = {}
                
                if label != 'O':
                    word_label[word] = label
                    list_word.append(word_label) 
                i+=1
                
    #La fonction renvoie finalement list_word.           
    return list_word 
 



#Même commentaire que pour la fonction fort_faible, sauf que là on classe les points faibles
def fort_faible2(text : list) :
    text = text.lower()
    text = re.split(r'[\.!?;,]| but | whereas | yet | althought | though | still | furthermore | moreover | even though | on the other hand | in spite of ', text)
    list_word = []
    
    for phrase in text :
        score = infer_onnx_model(phrase) 
        
        if score == 'negative' :
            prediction, model_output = model2.predict([phrase])
            pred = prediction
            
            i=0
            while i < len(pred[0]) :
                word, label = pred[0][i].popitem()
                word_label = {}
                    
                if label != 'O':
                    word_label[word] = label
                    list_word.append(word_label) 
            
                i+=1
                
    return list_word 








def group_point(word_list): #e code définit une fonction group_point qui prend en entrée une liste de mots (word_list).
    #La fonction initialise plusieurs variables à 0 (bedroom, bathroom, service, location, restaurant, price) qui serviront à compter le nombre d'occurrences de chaque mot dans la liste.
    bedroom = 0
    bathroom = 0
    service = 0
    location = 0
    restaurant = 0
    price = 0
    
    #La fonction définit également un dictionnaire vide (Dict_score) qui sera utilisé pour stocker le résultat final.
    Dict_score = {}
    
    #Ensuite, la fonction itère sur chaque mot de la liste (for i in word_list) et utilise la fonction next() avec l'itérable 
    #iter(i) pour récupérer la première clé du dictionnaire i. Le label de chaque mot est alors récupéré en accédant à la valeur associée à cette clé 
    for i in word_list :
        label = i[next(iter(i))]
        #Ensuite, la fonction utilise une série d'instructions if-elif pour incrémenter le compteur associé à chaque mot (bedroom, bathroom, service, etc.) 
        #en fonction de sa catégorie (bedroom, bathroom, service, etc.).
        if label == "bedroom" :
            bedroom = bedroom + 1
        elif label == "bathroom":
            bathroom = bathroom + 1
        elif label == "service":
            service = service + 1
        elif label == "location":
            location = location + 1
        elif label == "restaurant":
            restaurant = restaurant + 1
        elif label == "price":
            price = price + 1
    
    #Enfin, la fonction utilise la méthode update() du dictionnaire Dict_score pour ajouter les compteurs de chaque mot dans le dictionnaire résultat.
    Dict_score.update({"bedroom" : bedroom, "bathroom": bathroom, "service": service, "location": location, "restaurant": restaurant, "price": price}) 
    
    #La fonction retourne ce dictionnaire en fin d'exécution.
    return Dict_score  
  










#******************************************************Fast API*********************************************************
app = FastAPI()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# middlewares
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

@app.get('/')
async def root():
    return {'message': 'This is an API for sentiment analysis on hotel reviews'}



#application des fonctions :
@app.get('/ner')
async def ner_sentiment(x: str):
    fort = fort_faible(x)
    faible = fort_faible2(x)
    fort_by_category = group_point(fort)
    faible_by_category = group_point(faible)
    return "fort : ", fort , "faible : ", faible, "number of compliment by category : ", fort_by_category, "number of critique by category : ", faible_by_category
       




#********************************************************Notice*************************************************************

"""
1/run this in your anaconda's terminal (in anaconda prompt)
uvicorn main:app

2/copy the path and run it on google 
http://127.0.0.1:8000/docs

3/Click on the 2nd "GET" line

4/ Click on 'try it out'

5/ Writte your comment on the "White bar"

6/click on "execute"

7/look your result below.
"""















