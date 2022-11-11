import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
import pickle
from typing import List, Text
import configparser
import logging
from summa import keywords

try:
    from termcolor import colored
except:
    pass

try:
    import streamlit as st    
except ImportError:
    logging.info("Streamlit not installed")
config = configparser.ConfigParser()
try:
    config.read_file(open('paramconfig.cfg'))
except Exception:
    logging.warning("paramconfig file not found")
    st.info("Please place the paramconfig file in the same directory as app.py")


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def keywordExtraction(sdg:int,sdgdata:List[Text]):
    model_path = "docStore/sdg{}/".format(sdg)
    vectorizer = pickle.load(open(model_path+'vectorizer.pkl', 'rb'))
    tfidfmodel = pickle.load(open(model_path+'tfidfmodel.pkl', 'rb'))
    features = vectorizer.get_feature_names_out()
    tf_idf_vector=tfidfmodel.transform(vectorizer.transform(sdgdata))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    top_n = int(config.get('tfidf', 'TOP_N'))
    results=extract_topn_from_vector(features,sorted_items,top_n)
    keywords = [keyword for keyword in results]
    return keywords

def textrank(textdata, ratio = 0.1, words = 0):
    if words == 0:
        try:
            words = int(config.get('sdg','TOP_KEY'))
            results = keywords.keywords(textdata, words = words).split("\n")    
        except:
            logging.warning("paramconfig not found, running textrank with ratio")
            results = keywords.keywords(textdata, ratio= ratio).split("\n")
    else:
        results = keywords.keywords(textdata, words= words).split("\n")

    return results


