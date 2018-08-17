# LSTM_sentiment_analysis-
A Keras based LSTM network 

For this analysis, we will use a Long Short Term Memory (LSTM) network for sentiment analysis. LSTMs are form of Recurrent Neural Networks (RNN) that learn information from the immediate previous step. 

LSTMs are slightly different in that they can learn or remember both short and long. LSTMs can also be tuned to forget unnecessary information in a model, which can improve its ability to predict or classify data.  

As a quick test, we will use a LSTM network for sentiment analysis of tweets classifying whether a given tweet is positive or negative. 

First, we want to use a tokenizer that vectorizes the text (tweets) into a sequence of integers and then pad the sequences, which converts the sequences (of integers) into a two-dimensional numpy array. 

Our LSTM network is created using a Keras Sequential model with an embedding layer and fully connected dense layer, the latter which connects every node in a layer to every node in the preceding layer. 

Read Here: 



