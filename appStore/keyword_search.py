# set path
import glob, os, sys
from udfPreprocess.search import semantic_search
sys.path.append('../udfPreprocess')

#import helper
import udfPreprocess.docPreprocessing as pre
import udfPreprocess.cleaning as clean
from udfPreprocess.search import bm25_tokenizer, bm25TokenizeDoc, lexical_search
#import needed libraries
import seaborn as sns
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, CrossEncoder, util
# from keybert import KeyBERT
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd 
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import docx
from docx.shared import Inches
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE 
import logging
logger = logging.getLogger(__name__)
import tempfile
import sqlite3
import json
import configparser


def app():

    with st.container():
        st.markdown("<h1 style='text-align: center;  \
                      color: black;'> Search</h1>", 
                      unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=False):

        st.write(
            """     
            The *Keyword Search* app is an easy-to-use interface \ 
            built in Streamlit for doing keyword search in \
            policy document - developed by GIZ Data and the \
            Sustainable Development Solution Network.
            """)

        st.markdown("")
  
    
                  
    with st.sidebar:
        with open('sample/keywordexample.json','r') as json_file:
            keywordexample = json.load(json_file)
        
        genre = st.radio("Select Keyword Category", list(keywordexample.keys()))
        if genre == 'Food':
            keywordList = keywordexample['Food']
        elif genre == 'Climate':
            keywordList = keywordexample['Climate']
        elif genre == 'Social':
            keywordList = keywordexample['Social']
        elif genre == 'Nature':
            keywordList = keywordexample['Nature']
        elif genre == 'Implementation':
            keywordList = keywordexample['Implementation']
        else:
            keywordList = None
        
        searchtype = st.selectbox("Do you want to find exact macthes or similar meaning/context", ['Exact Matches', 'Similar context/meaning'])

        
    with st.container():
        if keywordList is not None:
            queryList = st.text_input("You selcted the {} category we will look for these keywords in document".format(genre),
                                    value="{}".format(keywordList))
        else:
            queryList = st.text_input("Please enter here your question and we will look \
                                     for an answer in the document OR enter the keyword you \
                                     are looking for and we will \
                                     we will look for similar context \
                                     in the document.",
                                    placeholder="Enter keyword here")

        if st.button("Find them"):

            if queryList == "":
                st.info("🤔 No keyword provided, if you dont have any, please try example sets from sidebar!")
                logging.warning("Terminated as no keyword provided")
            else:
                
                if 'docs' in st.session_state:
                    docs = st.session_state['docs']
                    paraList = st.session_state['paraList']
                   
                    if searchtype == 'Exact Matches':
                        queryList = list(queryList.split(","))
                        logging.info("performing lexical search")
                        tokenized_corpus = bm25TokenizeDoc(paraList)
                        # st.write(len(tokenized_corpus))
                        document_bm25 = BM25Okapi(tokenized_corpus)

                        with st.spinner("Performing Exact matching search (Lexical search) for you"):
                            st.markdown("##### Top few lexical search (BM25) hits #####")

                            for keyword in queryList:

                                bm25_hits = lexical_search(keyword,document_bm25)
                              
                                
                                counter = 0
                                for hit in bm25_hits:
                                    if hit['score'] > 0.00:
                                        counter += 1
                                        if counter == 1:
                                            st.markdown("###### Results for keyword: **{}** ######".format(keyword))
                                        # st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                                        st.write("\t {}: {}\t".format(counter, paraList[hit['corpus_id']].replace("\n", " ")))
                                        
                                st.markdown("---")
                                if counter == 0:
                                    st.write("No results found for '**{}**' ".format(keyword))
                    else:
                        logging.info("starting semantic search")
                        with st.spinner("Performing Similar/Contextual search"):
                            query = "Find {} related issues ?".format(queryList)
                            config = configparser.ConfigParser()
                            config.read_file(open('udfPreprocess/paramconfig.cfg'))
                            threshold = float(config.get('semantic_search','THRESHOLD'))
                            st.write(query)                          
                            semantic_hits = semantic_search(query,paraList)
                            st.markdown("##### Semantic search hits for {} related topics #####".format(queryList))

                            for i,queryhit in enumerate(semantic_hits):

                                # st.markdown("###### Results for query: **{}** ######".format(queryList[i]))
                                counter = 0
                                for hit in queryhit:
                                    counter += 1
                                    
                                                
                                    if hit['score'] > threshold:
                                    # st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                                        st.write("\t {}: \t {}".format(counter, paraList[hit['corpus_id']].replace("\n", " ")))

                                    # document.add_paragraph("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                                st.markdown("---")
                            # st.write(semantic_hits)

                          


                else:
                    st.info("🤔 No document found, please try to upload it at the sidebar!")
                    logging.warning("Terminated as no keyword provided")
                    
                
    