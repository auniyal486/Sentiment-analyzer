import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import re
from nltk.corpus import stopwords
import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint

stopWord=stopwords.words("english")
stopWord.remove('not')
num_words=102354 #num_words = len(data.word_index) + 1
Val_ACCURACY_THRESHOLD = 0.95
ACCURACY_THRESHOLD = 0.98

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > Val_ACCURACY_THRESHOLD):
            self.model.stop_training = True
        elif(logs.get('accuracy') > ACCURACY_THRESHOLD):
            self.model.stop_training = True

def cleaning_noise(String):
    tags = re.compile('<.*?>')
    string_without_tags = re.sub(tags, " ", String)
    string_without_url = re.sub(r'[\S]*\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?|[\S]*@gmail','',string_without_tags)
    string_without_punc=re.sub(r'[^a-zA-Z]',' ',string_without_url)
    noise_free_review=[i for i in string_without_punc.split() if i not in stopWord ]
    noise_free_review=' '.join(noise_free_review)
    return noise_free_review

def change_score(score):
    if(score<3):
        return 0
    elif score==3:
        return 1
    else:
        return 2

def restructure_df(reviews_df):
    df.dropna(inplace=True,subset=["ProductId","Text","Score"])
    df.drop("Id",axis=1,inplace=True)
    reviews_df=df[["Text","Score"]]
    reviews_df["Text"]=reviews_df["Text"].apply(cleaning_noise)
    reviews_df["Text"].apply(lambda x:len(x.split(" "))).hist()
    reviews_df=reviews_df[reviews_df["Text"].apply(lambda x:len(x.split(" "))in range(1,250))]
    corpus=reviews_df["Text"].values
    y_data=reviews_df["Score"].apply(change_score).values
    return corpus,y_data

def model_creation():
    model = Sequential()
    model.add(Embedding(num_words,8))
    model.add(LSTM(units = 50,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 3,activation='softmax'))
    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"]) 
    return model

if __name__=="__main__":
    df=pd.read_csv("Reviews.csv")
    corpus,y_data = restructure_df(df)
    data=Tokenizer(num_words=num_words)
    data.fit_on_texts(corpus)
    x_data=data.texts_to_sequences(corpus)
    x_data=pad_sequences(x_data,padding="pre")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.40, random_state = 0)
    model = model_creation()
    callbacks = myCallback()
    checkpointer = ModelCheckpoint(filepath="model.hdf5",monitor='val_accuracy',save_best_only=True)
    model.fit(x_train,y_train,epochs = 20,batch_size=256,validation_data=(x_test,y_test),callbacks=[callbacks,checkpointer])
    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())