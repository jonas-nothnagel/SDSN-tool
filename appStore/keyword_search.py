# set path
import glob, os, sys; 
sys.path.append('../utils')

import streamlit as st
import json
import logging
from utils.lexical_search import runLexicalPreprocessingPipeline, lexical_search
from utils.semantic_search import runSemanticPreprocessingPipeline, semantic_search

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
        with open('docStore/sample/keywordexample.json','r') as json_file:
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
        
        searchtype = st.selectbox("Do you want to find exact macthes or similar \
                                    meaning/context",
                                 ['Exact Matches', 'Similar context/meaning'])

        st.markdown("---")
    
    with st.container():
        if keywordList is not None:
            queryList = st.text_input("You selcted the {} category we \
                        will look for these keywords in document".format(genre),
                                    value="{}".format(keywordList))
        else:
            queryList = st.text_input("Please enter here your question and we \
                                        will look for an answer in the document\
                                        OR enter the keyword you are looking \
                                        for and we will we will look for similar\
                                        context in the document.",
                                    placeholder="Enter keyword here")
        
        if st.button("Find them"):

            if queryList == "":
                st.info("🤔 No keyword provided, if you dont have any, \
                                please try example sets from sidebar!")
                logging.warning("Terminated as no keyword provided")
            else:
                if 'filepath' in st.session_state:
                    
                    
                    if searchtype == 'Exact Matches':
                        allDocuments = runLexicalPreprocessingPipeline(
                                            st.session_state['filepath'],
                                            st.session_state['filename'])
                        logging.info("performing lexical search")
                        with st.spinner("Performing Exact matching search \
                                        (Lexical search) for you"):
                            st.markdown("##### Top few lexical search (TFIDF) hits #####")
                            lexical_search(queryList,allDocuments['documents'])
                    else:
                        allDocuments = runSemanticPreprocessingPipeline(
                                            st.session_state['filepath'],
                                            st.session_state['filename'])
                        
                        logging.info("starting semantic search")
                        with st.spinner("Performing Similar/Contextual search"):
                            semantic_search(queryList,allDocuments['documents'])

                else:
                    st.info("🤔 No document found, please try to upload it at the sidebar!")
                    logging.warning("Terminated as no document provided")
        


                    
