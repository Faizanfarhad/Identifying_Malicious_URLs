from keras import models
from sklearn.preprocessing import LabelEncoder
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences


model = models.load_model('saved Model/malicious_url_checker_model.keras')

vectorize = TextVectorization(
        max_tokens= 10000,
        standardize='lower_and_strip_punctuation',
        output_sequence_length=24,
        split='whitespace',
        output_mode='int',
        encoding='utf-8'
        )

url = ['"g00gle.com"']
vectorize.adapt(url)
encoded_url = vectorize(url)
padded_encoded_url = pad_sequences(encoded_url,maxlen=576,padding='post',value=0)
reshaped_padded_url = padded_encoded_url.reshape(-1,24,24)


pred = model.predict(reshaped_padded_url)

print(pred)