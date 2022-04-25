import sys
import csv
import multiprocessing



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

#Miscellanous libraries
from collections import Counter

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Deep learning libraries
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



path = '/content/drive/MyDrive/ColabNotebooks/'
path_news_preprocessed = path + 'news_cleaned_2018_02_13.preprocessed.jsonl'
path_news_shuffled = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.jsonl'
path_news_train = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.train.jsonl'
path_news_test = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.test.jsonl'
path_news_val = '/content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.val.jsonl'
path_news_embedded =  path + 'news_cleaned_2018_02_13.embedded.jsonl'


!shuf /content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.jsonl > \
      /content/drive/MyDrive/ColabNotebooks/news_cleaned_2018_02_13.preprocessed.shuffled.jsonl
      
      
count_lines = 0
with open(path_news_shuffled, 'r') as in_news:
    for line in tqdm(in_news):
        count_lines += 1
        
        
count_lines, int(count_lines * .8), int(count_lines * .1), \
    count_lines - (int(count_lines * 0.8) + int(count_lines * 0.1))
    
subdataset_size = int(count_lines * .05)

with open(path_news_shuffled, 'r') as in_news:
    with open(path_news_train, 'w') as out_train:
        with open(path_news_test, 'w') as out_test:
            with open(path_news_val, 'w') as out_val:
                for i, line in tqdm(enumerate(in_news)):
                    if i < count_lines * .1:
                        out_train.write(line)
                    #elif i < count_lines * .9:
                     #   out_test.write(line)
                    #else:
                    #   out_val.write(line)
                    
