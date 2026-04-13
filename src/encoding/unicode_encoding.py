from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
"""
    
    should be take a column and return the preprocessed  

    Returns:
        _type_: _description_
    """
class EncodingwithPadding:

    def encode_eachChar(string):
        url  =  []
        for c in string:
            url.append(ord(c))
        
        
        pad_len = 576
        url = np.array(url[:pad_len],dtype=np.int32)
        padded = np.pad(url,pad_width=(0,(pad_len - len(url))), mode='constant') #576 because 24*24 = 576
        url = padded.reshape((24,24))
        
        url = np.array(url,dtype=np.int32)
        return url


        

