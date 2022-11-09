import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle


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

def keyword_extraction(sdg:int,sdgdata):
    model_path = "docStore/sdg{}/".format(sdg)
    vectorizer = pickle.load(open(model_path+'vectorizer.pkl', 'rb'))
    tfidfmodel = pickle.load(open(model_path+'tfidfmodel.pkl', 'rb'))
    features = vectorizer.get_feature_names_out()
    


