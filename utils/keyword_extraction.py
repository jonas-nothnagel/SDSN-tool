import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import pickle
from typing import List, Text
import logging
from summa import keywords

try:
    import streamlit as st    
except ImportError:
    logging.info("Streamlit not installed")


def sort_coo(coo_matrix):
    """
    It takes Coordinate format scipy sparse matrix and extracts info from same.\
    1. https://kavita-ganesan.com/python-keyword-extraction/#.Y2-TFHbMJPb
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, top_n=10):
    """get the feature names and tf-idf score of top n items
    
    Params
    ---------
    feature_names: list of words from vectorizer
    sorted_items: tuple returned by sort_coo function defined in  \
    keyword_extraction.py
    topn: topn words to be extracted using tfidf

    Return
    ----------
    results: top extracted keywords

    """
    
    #use only topn items from vector
    sorted_items = sorted_items[:top_n]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


def tfidf_keyword(textdata:str, vectorizer, tfidfmodel, top_n):
    """
    TFIDF based keywords extraction
    
    Params
    ---------
    vectorizer: trained cont vectorizer model
    tfidfmodel: TFIDF Tranformer model
    top_n: Top N keywords to be extracted
    textdata: text data to which needs keyword extraction

    Return
    ----------
    keywords: top extracted keywords

    """
    features = vectorizer.get_feature_names_out()
    tf_idf_vector=tfidfmodel.transform(vectorizer.transform(textdata))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    results=extract_topn_from_vector(features,sorted_items,top_n)
    keywords = [keyword for keyword in results]
    return keywords

def keyword_extraction(sdg:int,sdgdata:List[Text], top_n:int=10):
    """
    TFIDF based keywords extraction
    
    Params
    ---------
    sdg: which sdg tfidf model to be used
    sdgdata: text data to which needs keyword extraction


    Return
    ----------
    keywords: top extracted keywords

    """
    model_path = "docStore/sdg{}/".format(sdg)
    vectorizer = pickle.load(open(model_path+'vectorizer.pkl', 'rb'))
    tfidfmodel = pickle.load(open(model_path+'tfidfmodel.pkl', 'rb'))
    features = vectorizer.get_feature_names_out()
    tf_idf_vector=tfidfmodel.transform(vectorizer.transform(sdgdata))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    top_n = top_n
    results=extract_topn_from_vector(features,sorted_items,top_n)
    keywords = [keyword for keyword in results]
    return keywords

@st.cache(allow_output_mutation=True)
def textrank(textdata:Text, ratio:float = 0.1, words:int = 0)->List[str]:
    """
    wrappper function to perform textrank, uses either ratio or wordcount to
    extract top keywords limited by words or ratio.
    1. https://github.com/summanlp/textrank/blob/master/summa/keywords.py

    Params
    --------
    textdata: text data to perform the textrank.
    ratio: float to limit the number of keywords as proportion of total token \
        in textdata
    words: number of keywords to be extracted. Takes priority over ratio if \
        Non zero. Howevr incase the pagerank returns lesser keywords than \
        compared to fix value then ratio is used.
    
    Return
    --------
    results: extracted keywords
    """
    if words == 0:
        logging.info("Textrank using defulat ratio value = 0.1, as no words limit given")
        results = keywords.keywords(textdata, ratio= ratio).split("\n")
    else:
        try:
            results = keywords.keywords(textdata, words= words).split("\n")
        except:
            results = keywords.keywords(textdata, ratio = ratio).split("\n")

    return results


