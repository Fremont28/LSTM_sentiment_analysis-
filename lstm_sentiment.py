import keras
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
from keras.layers import Dense

#import twitter dataset 
cols=['sentiment','id','date','query_string','user','text']
df=pd.read_csv("Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None,names=cols,
encoding='latin-1')
df.sentiment.value_counts() 
#drop columns 
df.drop(['id','date','query_string','user'],axis=1,inplace=True)

#length of the tweet 
df['pre_clean_len']=[len(t) for t in df.text]
df.info() 

tweets_sub=df[df.pre_clean_len>140]
tweets_sub.info() 
tweets_sub.sentiment.value_counts() 

#sentiment columns 
tweets_sub['sentiment'] = ['pos' if (x>0) else 'neg' for x in tweet_sub['sentiment']]
#clean the data 
tweets_sub['text']=tweets_sub['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

tokenizer=Tokenizer(nb_words=1000,lower=True,split=' ')
tokenizer.fit_on_texts(tweets_sub['text'].values)
tokenizer.word_index
X=tokenizer.texts_to_sequences(tweets_sub['text'].values) #converts each word to integer
X=pad_sequences(X) #converts sequences into a 2-d array

#lstm network 
embed_dim=128
lstm_out=200
batch_size=32

model=Sequential()
model.add(Embedding(2500,embed_dim,input_length=X.shape[1],dropout=0.27))
model.add(LSTM(lstm_out,dropout_U=0.35,dropout_W=0.25))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary 

#test the lstm model 
Y=pd.get_dummies(tweets_sub['sentiment'].values)
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.22)
#train the network 
model.fit(X_train,Y_train,batch_size,nb_epoch=1,verbose=3)
score,acc=model.evaluate(X_valid,Y_valid,verbose=2,batch_size=batch_size)
acc #69.33% accuracy 

