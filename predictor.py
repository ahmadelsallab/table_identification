# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import flask
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
import re
from autocorrect import spell

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
#vocab_file = 'vocab.npz'
#MODEL_NAME = 'keras_spell_trained_model.hdf5'
#ENC_MODEL_NAME = 'encoder_model.hdf5'
#DEC_MODEL_NAME = 'decoder_model.hdf5'
max_sent_lengths = [50, 100]
vocab_file = {}
vocab_to_int = {}
int_to_vocab = {}
max_sent_len = {}
min_sent_len = {}
num_decoder_tokens = {}
num_encoder_tokens = {}
max_encoder_seq_length = {}
max_decoder_seq_length = {}

for i in max_sent_lengths:
    vocab_file[i] = 'vocab-{}.npz'.format(i)
    
    vocab = np.load(file=vocab_file[i])
    vocab_to_int[i] = vocab['vocab_to_int'].item()
    int_to_vocab[i] = vocab['int_to_vocab'].item()
    max_sent_len[i] = vocab['max_sent_len']
    min_sent_len[i] = vocab['min_sent_len']
    input_characters = sorted(list(vocab_to_int))
    num_decoder_tokens[i] = num_encoder_tokens[i] = len(input_characters) #int(encoder_model.layers[0].input.shape[2])
    max_encoder_seq_length[i] = max_decoder_seq_length[i] = max_sent_len[i] - 1#max([len(txt) for txt in input_texts])

def clean_up_sentence(sentence, vocab):
    s = ''
    prev_char = ''
    for c in sentence.strip():
        if c not in vocab or (c == ' ' and prev_char == ' '):
            s += ''
        else:
            s += c
        prev_char = c
            
    return s

def vectorize_data(input_texts, max_encoder_seq_length, num_encoder_tokens, vocab_to_int):
    
    if(len(input_texts) > max_encoder_seq_length):
        input_texts = input_texts[:max_encoder_seq_length]
    
    '''Prepares the input text and targets into the proper seq2seq numpy arrays'''
    encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length),
    dtype='float32')
    
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text[:max_encoder_seq_length]):
            # c0..cn
            encoder_input_data[i, t] = vocab_to_int[char]
                
    return encoder_input_data

def word_spell_correct(decoded_sentence):
    corrected_decoded_sentence = ''
    special_chars = ['\\', '/', '-', '—' , ':', '[', ']', ',', '.', '\"', ';', '%', '~', '(', ')', '{', '}', '$', '&', '#', '☒', '■', '☐', '□', '☑', '@']
    for w in decoded_sentence.split(' '):
        if((len(re.findall(r'\d+', w))==0) and not (w in special_chars)):
            corrected_decoded_sentence += spell(w) + ' '
        else:
            corrected_decoded_sentence += w + ' '
    return corrected_decoded_sentence
    
def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, max_decoder_seq_length, vocab_to_int, int_to_vocab):
    
    #print(max_decoder_seq_length)
    # Encode the input as state vectors.
    encoder_outputs, h, c  = encoder_model.predict(input_seq)
    states_value = [h,c]
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab_to_int['\t']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    #print(input_seq)
    attention_density = []
    i = 0
    special_chars = ['\\', '/', '-', '—' , ':', '[', ']', ',', '.', '"', ';', '%', '~', '(', ')', '{', '}', '$', '#']
    #special_chars = []
    while not stop_condition:
        #print(target_seq)
        output_tokens, attention, h, c  = decoder_model.predict(
            [target_seq, encoder_outputs] + states_value)
        #print(attention.shape)
        attention_density.append(attention[0][0])# attention is max_sent_len x 1 since we have num_time_steps = 1 for the output
        # Sample a token
        #print(output_tokens.shape)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        #print(sampled_token_index)
        sampled_char = int_to_vocab[sampled_token_index]
        orig_char = int_to_vocab[int(input_seq[:,i][0])]
        
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
            #print('End', sampled_char, 'Len ', len(decoded_sentence), 'Max len ', max_decoder_seq_length)
            sampled_char = ''
        
        # Copy digits as it, since the spelling corrector is not good at digit corrections
        
        if(orig_char.isdigit() or orig_char in special_chars):
            decoded_sentence += orig_char            
        else:
            if(sampled_char.isdigit() or sampled_char in special_chars):
                decoded_sentence += ''
            else:
                decoded_sentence += sampled_char
        
        #decoded_sentence += sampled_char


        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
        
        i += 1
        if(i > 48):
            i = 0
    attention_density = np.array(attention_density)
    # Word level spell correct
    '''
    corrected_decoded_sentence = ''
    for w in decoded_sentence.split(' '):
        corrected_decoded_sentence += spell(w) + ' '
    decoded_sentence = corrected_decoded_sentence
    '''
    return decoded_sentence, attention_density

class ScoringService(object):
    #model_50 = None
    #model_100 = None
    encoder_model_50 = None
    decoder_model_50 = None
    encoder_model_100 = None
    decoder_model_100 = None
    
    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.encoder_model_50 is None:
            #cls.model = load_model(os.path.join(model_path, MODEL_NAME))
            #print(cls.model.summary())
            #cls.encoder_model = load_model(os.path.join(model_path, encoder_model-100.hdf5))
            #print(cls.encoder_model.summary())
            #cls.decoder_model = load_model(os.path.join(model_path, decoder_model-100.hdf5))
            #print(cls.decoder_model.summary())
            #cls.model_100 = load_model(os.path.join(model_path, 'model-100.hdf5'))
            #cls.model_50 = load_model(os.path.join(model_path, 'model-50.hdf5'))
            cls.encoder_model_100 = load_model(os.path.join(model_path, 'encoder_model-100.hdf5'))
            cls.encoder_model_50 = load_model(os.path.join(model_path, 'encoder_model-50.hdf5'))
            cls.decoder_model_100 = load_model(os.path.join(model_path, 'decoder_model-100.hdf5'))
            cls.decoder_model_50 = load_model(os.path.join(model_path, 'decoder_model-50.hdf5'))
            
        return cls.encoder_model_50, cls.decoder_model_50, cls.encoder_model_100, cls.decoder_model_100

    @classmethod
    def predict(cls, input_text):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        sess = K.get_session()
        with sess.graph.as_default():
            encoder_model_50, decoder_model_50, encoder_model_100, decoder_model_100 = cls.get_model()
            len_range = max_sent_lengths[-1] # Take the longest range
            for length in max_sent_lengths:
                if(len(input_text) < length):
                    len_range = length
                    break
                    
            input_text = clean_up_sentence(input_text, vocab_to_int[len_range])
            encoder_input_data = vectorize_data(input_texts=[input_text], max_encoder_seq_length=max_encoder_seq_length[len_range], 
                                                num_encoder_tokens=num_encoder_tokens[len_range], vocab_to_int=vocab_to_int[len_range])
            
            if len_range == 50:
                encoder_model = encoder_model_50
                decoder_model = decoder_model_50
            else:
                encoder_model = encoder_model_100
                decoder_model = decoder_model_100
                
            input_seq = encoder_input_data
            decoded_sentence, _ = decode_sequence(input_seq, encoder_model, decoder_model, 
                                                  num_decoder_tokens[len_range],  max_decoder_seq_length[len_range], 
                                                  vocab_to_int[len_range], int_to_vocab[len_range])
      
            corrected_sentence = word_spell_correct(input_text)
            #print('-Lenght = ', len_range)
            #print('Input sentence:', input_text)
            #print('Char Decoded sentence:', decoded_sentence)   
            #print('Word Decoded sentence:', corrected_sentence) 
            
            return corrected_sentence
# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        #data = flask.request.data.decode('utf-8')
        print(flask.request.data)
        data = json.loads(flask.request.data.decode('utf-8'))
        #s = StringIO.StringIO(data)
        #sent = data['sent']
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    #print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    for i, elem in enumerate(data):
        
        prediction = ScoringService.predict(elem['text'])
        elem.update({'corrected_text' : prediction})
        print(i+1)
    
    result = json.dumps(data, ensure_ascii=False)

    return flask.Response(response=result, status=200, mimetype='application/json')