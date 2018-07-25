import sys, os
sys.path.append(os.pardir)

os.environ["KERAS_BACKEND"]='tensorflow'
#import glob
import numpy as np
##
x_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed25/train_data_features_sorted.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_train_sort.npy')
y_train = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/x_test_sort.npy')
# print(len(x_train))
# 
x_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed25/test_data_features_sorted.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_train.npy')
y_test = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_scores_binary.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/t_test.npy')
# 
x_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed25/validation_data_features_sorted.npy')
y_valid = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_scores_binary.npy')
# 
#train_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/train_data_idx_binary_only.npy')
# test_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/test_data_idx_binary_only.npy')
# valid_idx = np.load('/home/louis/SharedWindows/edu_data/Preprocessed/validation_data_idx_binary_only.npy')

#print(len(train_idx))
#x_train = x_train[train_idx]
#y_train = y_train[train_idx]
# print(len(x_train))
# print(len(y_train))

# x_test = x_test[test_idx]
# #y_test = y_test[test_idx]
# 
# x_valid = x_valid[valid_idx]
#y_valid = y
#_valid[valid_idx]
#print(len(x_train))

# word_idx=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed/' + 'vocab_idx.npy')
#print(idx)
embWeights=np.load( '/home/louis/SharedWindows/edu_data/Preprocessed25/' + 'weights.npy')#np.load('/home/owner/デスクトップ/PythonFile/imdb/weights.npy')

#print(len(word_idx))
# print(y_test)


print('data loaded')

##

import keras
from keras.layers import Input, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import keras.backend as K
from keras.layers import Lambda, regularizers, Average

from keras.layers import RepeatVector, Permute, Multiply

from keras.models import Model
from keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense, Lambda, Masking
from keras.layers import merge, Layer, Activation, Dot, Concatenate, Flatten, Lambda

from keras.initializers import Identity,glorot_normal
from keras import regularizers

from keras import metrics

from keras.utils import plot_model

##

max_features = 20000
maxlen = 80 # cuts texts after this number of words
batch_size = 256


recursiveClass = GRU
wordRnnSize=100
sentenceRnnSize=100
dropWordEmb = 0.2
dropWordRnnOut = 0.2
dropSentenceRnnOut = 0.5

WORD_GRU_NUM = 30
dr = 0.5

eta = 1e-7
eta_dense = 1e-7

##
print('Build model...')
numSentencesPerDoc, numWordsPerSentence = x_train[0].shape[0], x_train[0].shape[1]
print("Number of sentences and words:")
print(numSentencesPerDoc, numWordsPerSentence)

vocabSize, embeddingSize = embWeights.shape[0], embWeights.shape[1]
print(vocabSize, embeddingSize)


x_in = Input( shape = ( numSentencesPerDoc, numWordsPerSentence ) , name='Input' )
embLayer = Embedding( input_dim=embWeights.shape[0], output_dim=embWeights.shape[1], weights=[embWeights]
                      ,mask_zero=False , trainable=False, embeddings_regularizer=regularizers.l2(0.0000001)
                      , input_length=numWordsPerSentence, name='Embedding' )

sent_vecs = []

extraDimLayer = Lambda(lambda x: K.expand_dims(x), name='extraDimForConvo')
squeezeSecondLayer = Lambda(lambda x: K.squeeze(x, 1), name='squeezeLayer')

 
biRnn_Layer = Bidirectional(GRU(WORD_GRU_NUM,  return_sequences=True, bias_regularizer=regularizers.l2(eta)
                           ,kernel_regularizer=regularizers.l2(eta),recurrent_regularizer=regularizers.l2(eta)
                           ,dropout=dr, recurrent_dropout=dropWordRnnOut, unroll=True), merge_mode='concat')
              
CONTEXT_DIM = 2*WORD_GRU_NUM                  
att_layer1 = Dense(CONTEXT_DIM, use_bias=True, activation='tanh')
att_layer2 = Dense(1, use_bias=False)

for i in range(numSentencesPerDoc):
    
    x_pop = Lambda(lambda x: x[:,i], output_shape=(numWordsPerSentence, ) , name='convert_shape_'+'sentence'+str(i))( x_in )
    
    emb = embLayer(x_pop)
    emb = Dropout(dropWordEmb)(emb)
    
   
    
    biRnn_word = biRnn_Layer(emb)#int(emb.shape[1])
    
    #CONTEXT_DIM = int(biRnn_word.shape[2])
    
    eij_ = att_layer1(biRnn_word)
    eij = att_layer2(eij_)
    eij_normalized = TimeDistributed(Activation('softmax'))(eij)
    #eij_permuted =Permute((2,1))(eij_normalized)

    sent_vec = Dot(axes=1)([eij_normalized, biRnn_word])#merge([eij_permuted, biRnn_word], mode = 'mul', name='word_attention_'+str(i))
    
    #sent_vec = Lambda(lambda x: k.sum(x, axis = 1))(weighted_input_) 
    #sent_vec = Lambda(lambda x: k.squeeze(x, axis = 1))(sent_vec)
    sent_vecs.append(sent_vec)

mergedSentVecs= Concatenate(axis = 1)(sent_vecs)

SENT_GRU_NUM = 50

biRnn_sent = Bidirectional(GRU(SENT_GRU_NUM,  return_sequences=True, bias_regularizer=regularizers.l2(eta)
                           ,kernel_regularizer=regularizers.l2(eta),recurrent_regularizer=regularizers.l2(eta)
                           ,dropout=dr, recurrent_dropout=dr, unroll=True), merge_mode='concat')(mergedSentVecs)

CONTEXT_DIM_SENT = int(biRnn_sent.shape[2])

eij_sent_ = Dense(CONTEXT_DIM_SENT, use_bias=True, activation='tanh')(biRnn_sent)
eij_sent = Dense(1, use_bias=False)(eij_sent_)
eij_sent_normalized = TimeDistributed(Activation('softmax'))(eij_sent)
#eij_sent_permuted =Permute((2,1))(eij_sent_normalized)

doc_vec_ = Dot(axes=1)([eij_sent_normalized, biRnn_sent])
doc_vec = squeezeSecondLayer(doc_vec_)
#doc_vec = Lambda(lambda x: k.sum(x, axis = 1))(doc_vec) 
#doc_vec = Lambda(lambda x: k.expand_dims(x, axis = 1))(doc_vec)

out = Dense(1,activation='sigmoid', use_bias=True)(doc_vec)


##

model = Model(inputs=[x_in], outputs=[out])

model.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
             metrics=['accuracy'])

print("Attention Model Build Complete")
##
name = './params_hiernet_adam_0725.hdf5'
save_model = keras.callbacks.ModelCheckpoint(name, monitor='val_loss', verbose=1
                                             , save_best_only=True, save_weights_only=True, mode='min', period=1)
##

history = model.fit(x_train, y_train, batch_size = batch_size, verbose=1, epochs=15 #epochs
                        ,validation_data=(x_valid, y_valid), shuffle=True, callbacks=[save_model])



##
# 
# model = Model(inputs=[x_in], outputs=[out, eij, eij_,eij_normalized, mergedPoolPerDoc])
# #adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# #model.compile(loss='binary_crossentropy',
# #              optimizer=adadelta,
# #              metrics=['accuracy'])
#          
# model.compile(loss='binary_crossentropy',
#               optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
#               metrics=['accuracy'])
#               
#               
# 
# print("Attention Model Build Complete")
##
s = model.evaluate(x_test, y_test)
print(s)
# ##
# weight_path = '/home/louis/SharedWindows/results/params_milnet_adam_0719_2_adjCNN_fixed_weights.hdf5'
# model.load_weights(weight_path)
# 
# 
# ##
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# 
# def plot_weights(w):
#     l = len(w)
#     fig = plt.figure(figsize=(30,16))
#     gs = gridspec.GridSpec(int(l/2+1),2)
#     
#     for i in range(l):
#         j = int( i/2 )
#         k = i%2
#         if k==0:
#             ax = plt.subplot(gs[j,k])
#             ax.hist( w[i].flatten() ,bins=100 ,range=(-0.5,0.5) )
#             #ax.set_title(title[i])
#         else:
#             ax = plt.subplot(gs[j,k])
#             ax.hist( w[i].flatten() ,bins=100 ,range=(-0.5,0.5) )
#             #ax.set_title(title[i])
#     plt.show()
#     #plt.close()
# 
# ##
# model.summary()
# ##
# plot_weights(model.get_layer('word_mat_convo_win_size_4').get_weights())
# ##
# #print(x_test[0])
# test = x_test[1:3].tolist()
# for l in range(len(test)):
#     for i in range(len(test[l])):
#         for j in range(len(test[l][i])):
#             if test[l][i][j] >= 61560:
#                 test[l][i][j] = 0
# 
# outputs = model.predict_on_batch(np.asarray(test))
# ##
# import time
# 
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
# 
# layer_name = 'word_mat_convo_win_size_3'
# filter_index = 0
# input_img = model.input
# 
# for filter_index in range(100):
#     print('Processing filter %d' % filter_index)
#     start_time = time.time()
# 
#     layer_output = layer_dict[layer_name].get_output_at(0)
#     loss = K.mean(layer_output[:,:,filter_index])#, axis=[0,1,2])
#     
#     
#     grads = K.gradients(loss, input_img)[0]
# 
#     iterate = K.function([input_img], [loss, grads])
# 
#     step = 1.0
# 
#     # we start from a gray image with some random noise
#     input_img_data = np.random.random((1, int(input_img[1]), int(input_img[2])))
#     
#     input_img_data = input_img_data * 10000
#     
#     for i in range(20):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
# 
#         print('Current loss value:', loss_value)
#         if loss_value <= 0.:
#             # some filters get stuck to 0, we can skip them
#             break
# 
#     # decode the resulting input image
#     if loss_value > 0:
#         img = deprocess_image(input_img_data[0])
#         kept_filters.append((img, loss_value))
#     end_time = time.time()
#     print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
#     
# ##
# from __future__ import print_function
# 
# import numpy as np
# import time
# 
# from keras.applications import vgg16
# from keras import backend as K
# img_width = 128
# img_height = 128
# 
# # the name of the layer we want to visualize
# # (see model definition at keras/applications/vgg16.py)
# layer_name = 'block5_conv1'
# 
# # util function to convert a tensor into a valid image
# 
# 
# def deprocess_image(x):
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + K.epsilon())
#     x *= 0.1
# 
#     # clip to [0, 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)
# 
#     # convert to RGB array
#     x *= 255
#     if K.image_data_format() == 'channels_first':
#         x = x.transpose((1, 2, 0))
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
# 
# 
# # build the VGG16 network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=False)
# print('Model loaded.')
# 
# model.summary()
# 
# # this is the placeholder for the input images
# input_img = model.input
# 
# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# 
# 
# def normalize(x):
#     # utility function to normalize a tensor by its L2 norm
#     return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
# 
# 
# kept_filters = []
# for filter_index in range(200):
#     # we only scan through the first 200 filters,
#     # but there are actually 512 of them
#     print('Processing filter %d' % filter_index)
#     start_time = time.time()
# 
#     # we build a loss function that maximizes the activation
#     # of the nth filter of the layer considered
#     layer_output = layer_dict[layer_name].output
#     if K.image_data_format() == 'channels_first':
#         loss = K.mean(layer_output[:, filter_index, :, :])
#     else:
#         loss = K.mean(layer_output[:, :, :, filter_index])
# 
#     # we compute the gradient of the input picture wrt this loss
#     grads = K.gradients(loss, input_img)[0]
# 
#     # normalization trick: we normalize the gradient
#     grads = normalize(grads)
# 
#     # this function returns the loss and grads given the input picture
#     iterate = K.function([input_img], [loss, grads])
# 
#     # step size for gradient ascent
#     step = 1.
# 
#     # we start from a gray image with some random noise
#     if K.image_data_format() == 'channels_first':
#         input_img_data = np.random.random((1, 3, img_width, img_height))
#     else:
#         input_img_data = np.random.random((1, img_width, img_height, 3))
#     input_img_data = (input_img_data - 0.5) * 20 + 128
# 
#     # we run gradient ascent for 20 steps
#     for i in range(20):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
# 
#         print('Current loss value:', loss_value)
#         if loss_value <= 0.:
#             # some filters get stuck to 0, we can skip them
#             break
# 
#     # decode the resulting input image
#     if loss_value > 0:
#         img = deprocess_image(input_img_data[0])
#         kept_filters.append((img, loss_value))
#     end_time = time.time()
#     print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
# 
