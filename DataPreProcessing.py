import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import string

import nltk
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from collections import  Counter

import nltk
from nltk.stem.isri import ISRIStemmer
from nltk import word_tokenize


df = pd.read_csv('arabic_tweets.csv')
# print(df.head())
df.drop(['Unnamed: 0'],inplace=True,axis=1)


##removing hashtags
def hashtag(text):
    pattern = r'#\w*'
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


##removing mentions
def mention(text):
    pattern = r'@\w*'
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


##removing puctuations
def puctuation(text):
    arabic_punctuations = r'''`÷×؛<>_()*&^%][ـ،/:"\؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


##removing diacritics
def diacritics(text):
    pattern = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    clean_text = re.sub(pattern, '', text)
    return clean_text


##removing digits
def digit(text):
    psttern = r'[~^0-9]'
    clean_text = ''.join([char for char in text if not char.isdigit()])
    return clean_text


##removing urls
def url(text):
    pattern = r'http[s]?://[^\s]+'
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


##removing emojis
def emoji(text):
    pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002500-\U00002BEF"  # chinese char
                         u"\U00002702-\U000027B0"
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         u"\U0001f926-\U0001f937"
                         u"\U00010000-\U0010ffff"
                         u"\u2640-\u2642"
                         u"\u2600-\u2B55"
                         u"\u200d"
                         u"\u23cf"
                         u"\u23e9"
                         u"\u231a"
                         u"\ufe0f"  # dingbats
                         u"\u3030"
                         "]+", re.UNICODE)
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


##removing extra_spaces
def extra_space(text):
    pattern = r'\s\s+'
    clean_text = re.sub(pattern, ' ', text)
    return clean_text


def english_words(text):
    pattern = r'[a-zA-Z]'
    clean_text = re.sub(pattern, '', text)
    return clean_text


##remove repeated characters
def repeated_char(text):
    return re.sub(r'(.)\1+', r'\1', text)



def clean_tweet(tweet):
    tweet = hashtag(tweet)
    tweet = mention(tweet)
    tweet = puctuation(tweet)
    tweet = digit(tweet)
    tweet = url(tweet)
    tweet = emoji(tweet)
    tweet = english_words(tweet)
    tweet = extra_space(tweet)
    tweet = diacritics(tweet)
    tweet = repeated_char(tweet)

    return tweet


##normalization
def normalize_char(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


##stemming
def stem(text):
    lst = []
    st = ISRIStemmer()

    for word in word_tokenize(text):
        lst.append(st.stem(word))
    return lst



df['tweets'] = df['tweets'].apply(lambda x:clean_tweet(x))
df['tweets'] = df['tweets'].apply(lambda x:normalize_char(x))

##label matching
df['dialect'] = df['dialect'].map({'IQ':0, 'LY':1, 'QA':2, 'PL':3, 'SY':4, 'TN':5, 'JO':6 , 'MA':7, 'SA':8, 'YE':9, 'DZ':10, 'EG':11, 'LB':12, 'KW':13, 'OM':14, 'SD':15, 'AE':16, 'BH':17})
df.to_csv('clean_data.csv')


# stemmed_df = df.copy()
#
# stemmed_df['tweets'] = df['tweets'].apply(lambda x:stem(x))
#
# ##lables
#
# stemmed_df['dialect'] = stemmed_df['dialect'].map({'IQ':0, 'LY':1, 'QA':2, 'PL':3, 'SY':4, 'TN':5, 'JO':6 , 'MA':7, 'SA':8, 'YE':9, 'DZ':10, 'EG':11, 'LB':12, 'KW':13, 'OM':14, 'SD':15, 'AE':16, 'BH':17})
#
#
# print(df['dialect'].value_counts())
#
#
#
# stemmed_df.to_csv('stemmed_data.csv')