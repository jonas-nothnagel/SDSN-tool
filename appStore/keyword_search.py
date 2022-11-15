# set path
import glob, os, sys; 
sys.path.append('../utils')

import streamlit as st
import json
import logging
from utils.lexical_search import runLexicalPreprocessingPipeline, lexical_search
from utils.semantic_search import runSemanticPreprocessingPipeline, semantic_search
from utils.checkconfig import getconfig

# Declare all the necessary variables
config = getconfig('paramconfig.cfg')
split_by = config.get('semantic_search','SPLIT_BY')
split_length = int(config.get('semantic_search','SPLIT_LENGTH'))
split_overlap = int(config.get('semantic_search','SPLIT_OVERLAP'))
split_respect_sentence_boundary = bool(int(config.get('semantic_search',
                                    'RESPECT_SENTENCE_BOUNDARY')))
remove_punc = bool(int(config.get('semantic_search','REMOVE_PUNC')))
embedding_model = config.get('semantic_search','RETRIEVER')
embedding_model_format = config.get('semantic_search','RETRIEVER_FORMAT')
embedding_layer = int(config.get('semantic_search','RETRIEVER_EMB_LAYER'))
retriever_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
reader_model = config.get('semantic_search','READER')
reader_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
lexical_split_by= config.get('lexical_search','SPLIT_BY')
lexical_split_length=int(config.get('lexical_search','SPLIT_LENGTH'))
lexical_split_overlap = int(config.get('lexical_search','SPLIT_OVERLAP'))
lexical_remove_punc = bool(int(config.get('lexical_search','REMOVE_PUNC')))
lexical_top_k=int(config.get('lexical_search','TOP_K'))

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
        st.write(""" The application allows its user to perform a keyword search\
             based on two options: a lexical (TFIDF) search and semantic  \
             bi-encoder search. The difference between both approaches is quite \
            straightforward; while the lexical search only displays paragraphs \
            in the document with exact matching results, the semantic search \
            shows paragraphs with meaningful connections (e.g., synonyms) based\
            on the context as well. The semantic search allows for a personalized\
            experience in using the application. Both methods employ a \
            probabilistic retrieval framework in its identification of relevant \
            paragraphs. By defualt the search is performed using 'Semantic Search'
            to find 'Exact/Lexical Matches' please tick the checkbox provided, which will \
            by pass semantic search.. Furthermore, the application allows the \
            user to search for pre-defined keywords from different thematic buckets\
            present in sidebar.""")

    
    with st.sidebar:
        with open('docStore/sample/keywordexample.json','r') as json_file:
            keywordexample = json.load(json_file)
        
        genre = st.radio("Select Keyword Category", list(keywordexample.keys()))
        if genre:
            keywordList = keywordexample[genre]
        else:
            keywordList = None
        
        st.markdown("---")
    
    with st.container():
        # if keywordList is not None:
        #     queryList = st.text_input("You selected the {} category we \
        #                 will look for these keywords in document".format(genre),
        #                             value="{}".format(keywordList))
        queryList = st.text_input("Please enter here your question and we \
                                    will look for an answer in the document\
                                    OR enter the keyword you are looking \
                                    for and we will we will look for similar\
                                    context in the document. You can select the \
                                    presets of keywords from sidebar.",
                                    value = "{}".format(keywordList))
        searchtype = st.checkbox("Show only Exact Matches")
        if st.button("Find them"):

            if queryList == "":
                st.info("🤔 No keyword provided, if you dont have any, \
                                please try example sets from sidebar!")
                logging.warning("Terminated as no keyword provided")
            else:
                if 'filepath' in st.session_state:
                    
                    
                    if searchtype:
                        allDocuments = runLexicalPreprocessingPipeline(
                                    file_name=st.session_state['filename'],
                                    file_path=st.session_state['filepath'],
                                    split_by=lexical_split_by,
                                    split_length=lexical_split_length,
                                    split_overlap=lexical_split_overlap,
                                    removePunc=lexical_remove_punc)
                        logging.info("performing lexical search")
                        with st.spinner("Performing Exact matching search \
                                        (Lexical search) for you"):
                            st.markdown("##### Top few lexical search (TFIDF) hits #####")
                            lexical_search(
                                query=queryList,
                                documents = allDocuments['documents'],
                                top_k = lexical_top_k )
                    else:
                        allDocuments = runSemanticPreprocessingPipeline(
                                            file_path= st.session_state['filepath'],
                                            file_name  = st.session_state['filename'],
                                            split_by=split_by,
                                            split_length= split_length,
                                            split_overlap=split_overlap,
                                            removePunc= remove_punc,
                        split_respect_sentence_boundary=split_respect_sentence_boundary)
                        if len(allDocuments['documents']) > 100:
                            warning_msg = ": This might take sometime, please sit back and relax."
                        else:
                            warning_msg = ""

                        logging.info("starting semantic search")
                        with st.spinner("Performing Similar/Contextual search{}".format(warning_msg)):
                            semantic_search(query = queryList, 
                            documents = allDocuments['documents'],
                            embedding_model=embedding_model, 
                            embedding_layer=embedding_layer,
                            embedding_model_format=embedding_model_format,
                            reader_model=reader_model,reader_top_k=reader_top_k,
                            retriever_top_k=retriever_top_k)

                else:
                    st.info("🤔 No document found, please try to upload it at the sidebar!")
                    logging.warning("Terminated as no document provided")
        


                    
