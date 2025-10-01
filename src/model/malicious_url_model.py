# from keras.utils import text_dataset_from_directory
import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from encoding.unicode_encoding import EncodingwithPadding
from sklearn.model_selection import train_test_split
from keras import metrics
import matplotlib.pyplot as plt 
from keras.applications import VGG16

def model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(24,24,1),padding="same"))
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding="same"))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding="same"))
    model.add(layers.MaxPool2D((2,2)))


    model.add(layers.Conv2D(128,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(128,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(128,(3,3),activation='relu',padding="same"))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.Conv2D(216,(3,3),activation='relu',padding="same"))
    model.add(layers.MaxPool2D((1,1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(576,activation='relu'))
    model.add(layers.Dense(576,activation='relu'))
    model.add(layers.Dense(4,activation='softmax'))
    return model
def pretrained_models(vgg16):
    conv_base = models.Sequential()
    conv_base.add(vgg16)
    conv_base.add(layers.Flatten())
    conv_base.add(layers.Dense(256,activation='relu'))
    conv_base.add(layers.Dense(4,activation='softmax'))
    return conv_base
# conv_model = model()
pretrained_model = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(32,32,3))
conv_model = pretrained_models(pretrained_model)

print(conv_model.summary())
df = pd.read_csv('src/Dataset/malicious_phish.csv')
encoded_data = EncodingwithPadding()

X ,y = encoded_data.encode(df)

X ,y = encoded_data.encode(df)

x_train,x_temp,y_train,y_temp =train_test_split(X,y, test_size=0.2,random_state=42,stratify=y)
'''⁡⁢⁣⁢ Note Startify ensures that in shuffle labels not change⁡ '''

x_val, x_test, y_val, y_test = train_test_split(x_temp,y_temp,
                                test_size=0.5 ,random_state=42, stratify=y_temp)


conv_model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['acc'])

history = conv_model.fit(x_train,y_train,
                epochs=20,
                batch_size=32,
                validation_data=(x_val,y_val))

conv_model.save_weights('model_weights.weights.h5')
conv_model.save('malicious_url_checker_model.keras')



history_dict = history.history

acc = history_dict['acc']
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']

results = conv_model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

epochs = range(1,len(acc)+1)

plt.plot(epochs,train_loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='validation Loss')
plt.title('Trianing and Vltidation Loss')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

val_acc = history_dict['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',label="validation Accuracy")
plt.title('Traning and Validation Accuracy ')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()