# set path
import glob, os, sys; 
sys.path.append('../utils')

import streamlit as st
import json
import logging
from utils.lexical_search import runLexicalPreprocessingPipeline, lexical_search
from utils.semantic_search import runSemanticPreprocessingPipeline, semantic_keywordsearch
from utils.checkconfig import getconfig
from utils.streamlitcheck import checkbox_without_preselect

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
embedding_dim  = int(config.get('semantic_search','EMBEDDING_DIM'))
max_seq_len = int(config.get('semantic_search','MAX_SEQ_LENGTH')) 
retriever_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
reader_model = config.get('semantic_search','READER')
reader_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
top_k_per_candidate = int(config.get('semantic_search','READER_TOP_K_PER_CANDIDATE')) 
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
        st.write("")
        st.write(""" The application allows its user to perform a keyword search\
             based on two options: a lexical ([TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf))\
             search and semantic [bi-encoder](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)\
            search. The difference between both \
            approaches is quite straightforward; while the lexical search only \
            displays paragraphs in the document with exact matching results, \
            the semantic search shows paragraphs with meaningful connections \
            (e.g., synonyms) based on the context as well. The semantic search \
            allows for a personalized experience in using the application. Both \
            methods employ a probabilistic retrieval framework in its identification\
            of relevant paragraphs. By defualt the search is performed using \
            'Semantic Search', and to find 'Exact/Lexical Matches' please tick the \
            checkbox provided which will by-pass semantic search. Furthermore,\
            the application allows the user to search for pre-defined keywords \
            from different thematic buckets present in sidebar.""")
        st.write("")
        st.write(""" The Exact Matches gives back top {} findings, and Semantic
        search provides with top {} answers.""".format(lexical_top_k, retriever_top_k))
        st.write("")
        st.write("")
        st.markdown("Some runtime metrics tested with cpu: Intel(R) Xeon(R) CPU @ 2.20GHz, memory: 13GB")
        col1,col2,col3= st.columns([2,4,4])
        with col1:
            st.caption("OCR File processing")
            # st.markdown('<div style="text-align: center;">50 sec</div>', unsafe_allow_html=True)
            st.write("50 sec")
           
        with col2:
            st.caption("Lexical Search on 200 paragraphs(~ 35 pages)")
            # st.markdown('<div style="text-align: center;">12 sec</div>', unsafe_allow_html=True)
            st.write("15 sec")
           
        with col3:
            st.caption("Semantic search on 200 paragraphs(~ 35 pages)")
            # st.markdown('<div style="text-align: center;">120 sec</div>', unsafe_allow_html=True)
            st.write("120 sec(including emebedding creation)")
 
    with st.sidebar:
        with open('docStore/sample/keywordexample.json','r') as json_file:
            keywordexample = json.load(json_file)
        
        # genre = st.radio("Select Keyword Category", list(keywordexample.keys()))
        st.caption("Select Keyword Category")
        genre = checkbox_without_preselect(list(keywordexample.keys()))
        if genre:
            keywordList = keywordexample[genre]
        else:
            keywordList = None
        
        st.markdown("---")
    
    with st.container():
        type_hinting = "Please enter here your question and we \
                        will look for an answer in the document\
                        OR enter the keyword you are looking \
                        for and we will we will look for similar\
                        context in the document. If dont have anything,\
                        try the presets of keywords from sidebar. "
        if keywordList is not None:
        #     queryList = st.text_input("You selected the {} category we \
        #                 will look for these keywords in document".format(genre)
        #                             value="{}".format(keywordList))
            queryList = st.text_input(type_hinting,
                                        value = "{}".format(keywordList))
        else:
             queryList = st.text_input(type_hinting,
                                       placeholder="Enter keyword/query here")

        searchtype = st.checkbox("Show only Exact Matches")
        if st.button("Find them"):

            if queryList == "":
                st.info("🤔 No keyword provided, if you dont have any, \
                                please try example sets from sidebar!")
                logging.warning("Terminated as no keyword provided")
            else:
                if 'filepath' in st.session_state:
                      
                    if searchtype:
                        all_documents = runLexicalPreprocessingPipeline(
                                    file_name=st.session_state['filename'],
                                    file_path=st.session_state['filepath'],
                                    split_by=lexical_split_by,
                                    split_length=lexical_split_length,
                                    split_overlap=lexical_split_overlap,
                                    remove_punc=lexical_remove_punc)
                        logging.info("performing lexical search")
                        with st.spinner("Performing Exact matching search \
                                        (Lexical search) for you"):
                            lexical_search(query=queryList,
                        documents = all_documents['documents'],
                                top_k = lexical_top_k )
                    else:
                        all_documents = runSemanticPreprocessingPipeline(
                                            file_path= st.session_state['filepath'],
                                            file_name  = st.session_state['filename'],
                                            split_by=split_by,
                                            split_length= split_length,
                                            split_overlap=split_overlap,
                                            remove_punc= remove_punc,
                        split_respect_sentence_boundary=split_respect_sentence_boundary)
                        if len(all_documents['documents']) > 100:
                            warning_msg = ": This might take sometime, please sit back and relax."
                        else:
                            warning_msg = ""

                        logging.info("starting semantic search")
                        with st.spinner("Performing Similar/Contextual search{}".format(warning_msg)):
                            semantic_keywordsearch(query = queryList, 
                            documents = all_documents['documents'],
                            embedding_model=embedding_model, 
                            embedding_layer=embedding_layer,
                            embedding_model_format=embedding_model_format,
                            reader_model=reader_model,reader_top_k=reader_top_k,
                            retriever_top_k=retriever_top_k, embedding_dim=embedding_dim,
                            max_seq_len=max_seq_len,
                            top_k_per_candidate = top_k_per_candidate)

                else:
                    st.info("🤔 No document found, please try to upload it at the sidebar!")
                    logging.warning("Terminated as no document provided")
        


                    
