# from __future__ import print_function
# import numpy as np
# from model import createHierarchicalAttentionModel
# np.random.seed(1337)
# 
# from keras.preprocessing import sequence
# from keras.datasets import imdb
# 
# max_features = 20000
# maxlen = 80 # cuts texts after this number of words
# batch_size = 32
# 
# print('loading data...')
# 
# (x_train, y_train), (x_test, y_test) = imdb.load_data(nb_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test),'test sequences')
# 
# print('Pad sequences (sample x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# 
# x_train = np.expand_dims(x_train, axis=1)
# x_test = np.expand_dims(x_test, axis=1)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# 
# print('Build model...')
# model, modelAttEval=createHierarchicalAttentionModel(maxlen, embeddingSize=200, vocabSize=max_features)
# 
# print('Train...')
# model.fit(x_train, y_train, batch_size=batch_size, nb_epoch = 2, validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
# 
# print('test score:', score)
# print('test accuracy:', acc)

'''Trains a Hierarchical Attention Model on the IMDB sentiment classification task.
Modified from keras' examples/imbd_lstm.py.
'''
from __future__ import print_function
import numpy as np
#from model import createHierarchicalAttentionModel
np.random.seed(1337)  # for reproducibility

import os
os.environ['KERAS_BACKEND']='tensorflow'


from keras.preprocessing import sequence
from keras.datasets import imdb


from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.layers import merge, Layer, Dot, Concatenate
from keras.layers import Lambda, regularizers, Average, Multiply, Add

from keras import callbacks

from keras import backend as k

max_features = 20000
maxlen = 80 # cuts texts after this number of words
batch_size = 256


recursiveClass = GRU
wordRnnSize=100
sentenceRnnSize=100
dropWordEmb = 0.2
dropWordRnnOut = 0.2
dropSentenceRnnOut = 0.5
dr = 0.5

eta = 1e-7
eta_dense = 1e-7

##

# embWeights=np.load('/home/louis/SharedWindows/weights.npy')
# idx=np.load('/home/louis/SharedWindows/index.npy')
# embWeights = embWeights[idx]

print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

x_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_features.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_train_sort.npy')
y_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_test_sort.npy')
print(len(x_train))

x_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_features.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_train.npy')
y_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_test.npy')

x_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_features.npy')
y_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_scores_binary.npy')

train_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_idx_binary_only.npy')
test_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_idx_binary_only.npy')
valid_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_idx_binary_only.npy')

#print(len(train_idx))
x_train = x_train[train_idx]
#y_train = y_train[train_idx]
print(len(x_train))
print(len(y_train))

x_test = x_test[test_idx]
#y_test = y_test[test_idx]

x_valid = x_valid[valid_idx]
#y_valid = y_valid[valid_idx]
#print(len(x_train))

word_idx=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed/' + 'vocab_idx.npy')
#print(idx)
embWeights=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed/' + 'weights.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/weights.npy')

#print(len(word_idx))
print(y_test)

print('data loaded')

##
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

features = x_valid
print(len(features))
for i in range(len(features)):
    #print(len(features[i]))
    if len(features[i])!=40:
            print('i:'+str(i)+'   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
            print(len(features[i]))
            x_valid[i].append(x_valid[i][-1])
    for j in range(len(features[i])):
        
        #print(len(features[i][j]))
        if len(features[i][j])!= 15:
            print('i:'+str(i)+'j:'+str(j)+'   ************************************************************')

x_valid=np.asarray(x_valid.tolist())
##
print('Build model...')
numSentencesPerDoc, numWordsPerSentence = x_train[0].shape[0], x_train[0].shape[1]
print("Number of sentences and words:")
print(numSentencesPerDoc, numWordsPerSentence)

vocabSize, embeddingSize = embWeights.shape[0], embWeights.shape[1]
print(vocabSize, embeddingSize)


x_in = Input( shape = ( numSentencesPerDoc, numWordsPerSentence ) , name='Input' )
embLayer = Embedding( input_dim=embWeights.shape[0], output_dim=embWeights.shape[1], weights=[embWeights]
                      ,mask_zero=True , trainable=True, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=numWordsPerSentence, name='Embedding' )

sent_vecs = []

extraDimLayer = Lambda(lambda x: K.expand_dims(x), name='extraDimForConvo')
squeezeSecondLayer = Lambda(lambda x: K.squeeze(x, 1), name='squeezeThirdLayer')

for i in range(numSentencesPerDoc):
    
    x_pop = Lambda(lambda x: x[:,i], output_shape=(numWordsPerSentence, ) , name='convert_shape_'+'sentence'+str(i))( x_in )
    
    emb = embLayer(x_pop)
    emb = Dropout(dropWordEmb)(emb)
    
    WORD_GRU_NUM = 30
    
    biRnn_word = Bidirectional(GRU(WORD_GRU_NUM,  return_sequences=True, bias_regularizer=regularizers.l2(eta)
                           ,kernel_regularizer=regularizers.l2(eta),recurrent_regularizer=regularizers.l2(eta)
                           ,dropout=dr, recurrent_dropout=dropWordRnnOut, unroll=True), merge_mode='concat')(emb)#int(emb.shape[1])
    
    CONTEXT_DIM = int(biRnn_word.shape[2])
    
    eij = Dense(CONTEXT_DIM, use_bias=True, activation='tanh')(biRnn_word)
    eij = Dense(CONTEXT_DIM, use_bias=False, activation='softmax')(eij)

    weighted_input_ = merge([eij, biRnn_word], mode = 'mul', name='word_attention_'+str(i))
    
    sent_vec = Lambda(lambda x: k.sum(x, axis = 1))(weighted_input_) 
    sent_vec = Lambda(lambda x: k.expand_dims(x, axis = 1))(sent_vec)
    sent_vecs.append(sent_vec)

mergedSentVecs= Concatenate(axis = 1)(sent_vecs)

SENT_GRU_NUM = 50

biRnn_sent = Bidirectional(GRU(SENT_GRU_NUM,  return_sequences=True, bias_regularizer=regularizers.l2(eta)
                           ,kernel_regularizer=regularizers.l2(eta),recurrent_regularizer=regularizers.l2(eta)
                           ,dropout=dr, recurrent_dropout=dr, unroll=True), merge_mode='concat')(mergedSentVecs)

CONTEXT_DIM_SENT = int(biRnn_sent.shape[2])

eij_sent = Dense(CONTEXT_DIM_SENT, use_bias=True, activation='tanh')(biRnn_sent)
eij_sent = Dense(CONTEXT_DIM_SENT, use_bias=False, activation='softmax')(eij_sent)

weighted_input_sent = merge([eij_sent, biRnn_sent], mode = 'mul', name='sentence_attention')

doc_vec = Lambda(lambda x: k.sum(x, axis = 1))(weighted_input_sent) 
#doc_vec = Lambda(lambda x: k.expand_dims(x, axis = 1))(doc_vec)

out = Dense(1,activation='sigmoid', use_bias=True)(doc_vec)

##
model = Model(input=[x_in], output=[out])
model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

##
'''



    wordsInputs = Input(shape=(maxSeq,), dtype='int32', name='words_input')
    if embWeights is None:
        emb = Embedding(vocabSize, embeddingSize, mask_zero=True, embedding_regularizer = 1e-7)(wordsInputs)
    else:
        emb = Embedding(embWeights.shape[0], embWeights.shape[1], mask_zero=True, weights=[embWeights], trainable=False)(wordsInputs)
    if dropWordEmb != 0.0:
        emb = Dropout(dropWordEmb)(emb)
    wordRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(emb)
    if dropWordRnnOut  > 0.0:
        wordRnn = Dropout(dropWordRnnOut)(wordRnn)
    attention = AttentionLayer()(wordRnn)
    sentenceEmb = merge([wordRnn, attention], mode=lambda x:x[1]*x[0], output_shape=lambda x:x[0])
    sentenceEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]))(sentenceEmb)
    modelSentence = Model(wordsInputs, sentenceEmb)
    modelSentAttention = Model(wordsInputs, attention)
    
    
    documentInputs = Input(shape=(None,maxSeq), dtype='int32', name='document_input')
    sentenceMasking = Masking(mask_value=0)(documentInputs)
    sentenceEmbbeding = TimeDistributed(modelSentence)(documentInputs)
    sentenceAttention = TimeDistributed(modelSentAttention)(sentenceMasking)
    sentenceRnn = Bidirectional(recursiveClass(wordRnnSize, return_sequences=True), merge_mode='concat')(sentenceEmbbeding)
    if dropSentenceRnnOut > 0.0:
        sentenceRnn = Dropout(dropSentenceRnnOut)(sentenceRnn)
    attentionSent = AttentionLayer()(sentenceRnn)
    documentEmb = merge([sentenceRnn, attentionSent], mode=lambda x:x[1]*x[0], output_shape=lambda x:x[0])
    documentEmb = Lambda(lambda x:K.sum(x, axis=1), output_shape=lambda x:(x[0],x[2]), name="att2")(documentEmb)
    documentOut = Dense(1, activation="sigmoid", name="documentOut")(documentEmb)
    
    
    model = Model(input=[documentInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    
    modelAttentionEv = Model(inputs=[documentInputs], outputs=[documentOut,  sentenceAttention, attentionSent])
    modelAttentionEv.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])'''
##
print('Train...')
save_model = callbacks.ModelCheckpoint('./params_Hier_0717_3.hdf5', monitor='val_loss', verbose=1
                                             , save_best_only=True, save_weights_only=True, mode='min', period=1)
                                             
model.fit(x_train, y_train, batch_size=batch_size, epochs=50)
          #validation_data=(x_valid, y_valid), callbacks = [save_model])
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
