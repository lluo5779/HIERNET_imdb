import numpy as np
import pandas as pd
import _pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from tensorflow.python import keras
 
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.layers import Embedding, Bidirectional, Merge

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.layers import Lambda, regularizers, Average
# 
# MAX_SENT_LENGTH = 100
# MAX_SENTS = 15
# MAX_NB_WORDS = 20000
# EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2
# 
# def clean_str(string):
#     """
#     Tokenization/string cleaning for dataset
#     Every dataset is lower cased except
#     """
#     string = re.sub(r"\\", "", string)    
#     string = re.sub(r"\'", "", string)    
#     string = re.sub(r"\"", "", string)    
#     return string.strip().lower()
# 
# data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
# print(data_train.shape)
# 
# from nltk import tokenize
# 
# reviews = []
# labels = []
# texts = []
# 
# for idx in range(data_train.review.shape[0]):
#     text = BeautifulSoup(data_train.review[idx], "lxml")
#     text = clean_str(text.get_text().encode('ascii','ignore').decode('utf-8'))
#     texts.append(text)
#     sentences = tokenize.sent_tokenize(text)
#     reviews.append(sentences)
#     
#     labels.append(data_train.sentiment[idx])
# 
# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)
# 
# data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
# 
# for i, sentences in enumerate(reviews):
#     for j, sent in enumerate(sentences):
#         if j< MAX_SENTS:
#             wordTokens = text_to_word_sequence(sent)
#             k=0
#             for _, word in enumerate(wordTokens):
#                 if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
#                     data[i,j,k] = tokenizer.word_index[word]
#                     k=k+1                    
#                     
# word_index = tokenizer.word_index
# print('Total %s unique tokens.' % len(word_index))
# 
# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
# 
# 
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# 
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]
# 
# print('Number of positive and negative reviews in traing and validation set')
# print(y_train.sum(axis=0))
# print(y_val.sum(axis=0))
# 
# GLOVE_DIR = "" #"/ext/home/analyst/Testground/data/glove"
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# 
# print('Total %s word vectors.' % len(embeddings_index))
# 
# input("STOP in the name of love")
# 
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector


## Hierachical Attention Net

# building Hierachical Attention network


##GLOVE

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

data_train = pd.read_csv('/home/louis/Documents/Uchida_OU/HIERNET/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

from nltk import tokenize

reviews = []
labels = []
texts = []

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")
    text = clean_str(text.get_text().encode('ascii','ignore').decode('utf-8'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    
    labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1                    
                    
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
x_test = labels[:-nb_validation_samples]
t_train = data[-nb_validation_samples:]
t_test = labels[-nb_validation_samples:]

GLOVE_DIR = "/home/louis/Documents/Uchida_OU/HIERNET/" #"/ext/home/analyst/Testground/data/glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)



## OU Data Source
dir = '/home/louis/SharedWindows/'

x_train = np.load(dir + 'x_train_sort.npy')
x_test = np.load(dir + 'x_test_sort.npy')
t_train = np.load(dir + 't_train.npy')
t_test = np.load(dir + 't_test.npy')

weights=np.load(dir + 'weights.npy')
idx=np.load(dir + 'index.npy')
weights = weights[idx]

max_text_num, max_text_len = x_train[0].shape[0], x_train[0].shape[1]

embedding_layer = Embedding(input_dim=50, output_dim=len(weights['apple']), weights=[weights]
                      ,mask_zero=True , trainable=True, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=max_text_len, name='Embedding' )
##

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')#.decode('utf-8')
        print(weighted_input)
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

sentence_input = Input(shape=(max_text_len,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(max_text_num, max_text_len ), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
              
#save model to png file
from keras.utils import plot_model
plot_model( model, to_file='model.png' )

#モデルを保存せず直接可視化
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG( model_to_dot( model ).create( prog='dot', format='svg' ) )

model.summary()

#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=8)
save_model = keras.callbacks.ModelCheckpoint('./params.hdf5', monitor='val_loss', verbose=1
                                             , save_best_only=True, save_weights_only=True, mode='min', period=1)
#model.load_weights( '../LSTM1_NN1/params.hdf5', by_name=True )

print("model fitting - Hierachical attention network")
          
history = model.fit(x_train, t_train, batch_size=256, verbose=1, epochs=100
                    ,validation_split=0.2, shuffle=True, callbacks=[save_model])
                    
#評価
score = model.evaluate(x_test,t_test,batch_size=256)
print('loss='+str(score[0])+' acc='+str(score[1]))

#学習の経過を保存
np.save('./result_acc.npy', history.history['acc'])
np.save('./result_loss.npy', history.history['loss'])
np.save('./result_val_acc.npy', history.history['val_acc'])
np.save('./result_val_loss.npy', history.history['val_loss'])
