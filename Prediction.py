#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pickle import load


# In[9]:


features = load(open('features.pkl', 'rb'))
tokenizer = load(open('tokenizer.pkl', 'rb'))
corpus = load(open('corpus.pkl', 'rb'))


# In[18]:


max_length = 34
voc_size = len(tokenizer.word_index) + 1


# In[19]:



# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    img_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    # load the photo
    image = load_img(filename, target_size=(224, 224, 3))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # get features
    feature = img_model.predict(image, verbose=0)
    return feature





# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        
        # predict next word
        
        sequence = pad_sequences([sequence], maxlen=max_length)[0]
    
        X_train = [np.array(photo.reshape(1,2048)),np.array(sequence.reshape(1,34))]
        
        yhat = model.predict(X_train, verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# In[20]:


# load the model
model = keras.models.load_model("model_1.h5")
model._make_predict_function()


# In[21]:


def caption_this_image(image):
    photo=extract_features(image)
    description = generate_desc(model, tokenizer, photo, max_length)
    query = description
    stopwords = ['startseq','endseq']
    querywords = query.split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    return result



