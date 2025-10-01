from sklearn.preprocessing import LabelEncoder
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

class EncodingwithPadding:

    def encode(self,df):
        
        le = LabelEncoder()
        vectorize = TextVectorization(
        max_tokens= 10000,
        standardize='lower_and_strip_punctuation',
        output_sequence_length=24,
        split='whitespace',
        output_mode='int',
        encoding='utf-8'
        )

        vectorize.adapt(df['url'].to_list())
        encoded_url = vectorize(df['url'].to_list())
        
        df['type'] = le.fit_transform(df['type'])
        labeled_url = df['type']
        
        encoded_url = encoded_url.numpy()
        padded_encoded_url = pad_sequences(encoded_url,maxlen=3072,padding='post',value=0)
        reshaped_padded_url = padded_encoded_url.reshape(-1,32,32,3)

        return reshaped_padded_url,labeled_url


