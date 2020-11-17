import pandas as pd
import numpy as np

import os
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


def basic_clean(string):
    # lowercase all letters
    string = string.lower()
    # normalize unicode characters
    string = unicodedata.normalize('NFKD', string)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
    # replace everything that isn't letters, numbers, 
    # whitespace, or single quotes
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def tokenize(string):
    # creates tokenizer object
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # returns the string tokeized
    return tokenizer.tokenize(string, return_str=True)

def stem(text):
    # creates the stemming object
    ps = nltk.porter.PorterStemmer()
    # creates variable stem that reads all words in text split into a list as a list
    stems = [ps.stem(word) for word in text.split()]
    # creates variable to join all words from previous list with a space as one string
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(text):
    # create the lemmatization object
    wnl = nltk.stem.WordNetLemmatizer()
    # splits words in text to a list, then lemmatizes each word in the list
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    # joins each word as a single string with a space between the words
    text = ' '.join(lemmas)
    return text

def remove_stopwords(text, extra_words=[], exclude_words=[]):
    # creates list of stop words
    stopword_list = stopwords.words('english')
    
    # remove 'exclude_words' from stopword_list to keep these in the text.
    stopword_list = set(stopword_list) - set(exclude_words)

    # add 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))
    
    # split words in string.
    words = text.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords

def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    df = df.rename(columns={column:'original'})
    df['clean'] = df.original.apply(basic_clean)\
        .apply(tokenize)\
        .apply(remove_stopwords,
               extra_words=extra_words, 
               exclude_words=exclude_words)
    df['stemmed'] = df.clean.apply(stem)
    df['lemmatized'] = df.clean.apply(lemmatize)
    return df