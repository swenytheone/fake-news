import sys
import csv
import multiprocessing
from statistics import mean



import ujson
import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Conv1D, GlobalMaxPooling1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model, Model
import tensorflow as tf
import gensim
from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
from tqdm import tqdm 
import seaborn as sns
import numpy as np

#Basic libraries
import pandas as pd 
import numpy as np 

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 5]
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#NLTK libraries
import nltk
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
import string


# Machine Learning libraries
import sklearn 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 

#Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#Miscellanous libraries
from collections import Counter

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Deep learning libraries
from keras.preprocessing import text,sequence

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import gensim
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


data_train = pd.read_json('/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.train.jsonl', lines= True)
data_train.drop(['id'], inplace=True, axis=1)
data_train['new_content'] = data_train['content'].apply(' '.join)
data_train['new_content'][0]

data_train['label'].value_counts()


def normalize(data):
    normalized = []
    for i in data:
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

data_train['new_content'] = normalize(data_train['new_content'])

text_len1=data_train[data_train['label']==0]['new_content'].str.split().map(lambda x: len(x))
text_len=data_train[data_train['label']==1]['new_content'].str.split().map(lambda x: len(x))


max_len = int(mean(text_len) + mean(text_len1)) / 2 +1

texts = ' '.join(data_train['new_content'])
string = texts.split(" ")

features = data_train['new_content']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=42)

max_vocab = 10000
max_len = int(max_len)
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)
# tokenize the text into vectors 
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=max_len)

checkpoint_path = 'mdl_wts.hdf5'

mcp_save = ModelCheckpoint(
    checkpoint_path, 
    #save_best_only=True,
    save_weights_only=True, 
    monitor='val_loss', 
    mode='max')
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, #0.5
    patience=6, 
    verbose=1,
    mode='max')
tensor_board = TensorBoard(
    log_dir='./Graph', 
    histogram_freq=0, 
    write_graph=True, 
    write_images=True)
callbacks = [mcp_save, reduce_lr_loss]


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam'),
              #optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=128, shuffle=True, callbacks=[mcp_save, early_stop , reduce_lr_loss]) #early_stop


history = model.fit(X_train, y_train, epochs=100, validation_split=0.25, batch_size=256, shuffle=True, callbacks=[mcp_save , early_stop, reduce_lr_loss]) #early_stop

model.load_weights(checkpoint_path)



history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

plt.figure(figsize=(10,8))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()

plt.figure(figsize=(10,8))
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0.5,1))
plt.show()


model.evaluate(X_test, y_test)
pred = model.predict(X_test)
binary_predictions = []

for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0) 

print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
print('Precision on testing set:', precision_score(binary_predictions, y_test))
print('Recall on testing set:', recall_score(binary_predictions, y_test))

matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
plt.figure(figsize=(13, 8))
ax= plt.subplot()
sns.heatmap(matrix, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted Labels', size=20)
ax.set_ylabel('True Labels', size=20)
ax.set_title('Confusion Matrix', size=20) 
ax.xaxis.set_ticklabels([0,1], size=15)
ax.yaxis.set_ticklabels([0,1], size=15)