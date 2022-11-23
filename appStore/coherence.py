# set path
import glob, os, sys; 
sys.path.append('../utils')

import streamlit as st
import ast
import logging
from utils.ndc_explorer import countrySpecificCCA, countrySpecificCCM
from utils.checkconfig import getconfig
from utils.semantic_search import runSemanticPreprocessingPipeline,process_semantic_output
from utils.semantic_search import semanticSearchPipeline, runSemanticPipeline
from st_aggrid import AgGrid
from st_aggrid.shared import ColumnsAutoSizeMode

# Reading data and Declaring necessary variables
with open('docStore/ndcs/countryList.txt') as dfile:
    countryList = dfile.read()
countryList = ast.literal_eval(countryList)
countrynames = list(countryList.keys())
    
with open('docStore/ndcs/cca.txt', encoding='utf-8', errors='ignore') as dfile:
    cca_sent = dfile.read()
cca_sent = ast.literal_eval(cca_sent)
            
with open('docStore/ndcs/ccm.txt', encoding='utf-8', errors='ignore') as dfile:
    ccm_sent = dfile.read()
ccm_sent = ast.literal_eval(ccm_sent)

config = getconfig('paramconfig.cfg')
split_by = config.get('coherence','SPLIT_BY')
split_length = int(config.get('coherence','SPLIT_LENGTH'))
split_overlap = int(config.get('coherence','SPLIT_OVERLAP'))
split_respect_sentence_boundary = bool(int(config.get('coherence',
                                    'RESPECT_SENTENCE_BOUNDARY')))
remove_punc = bool(int(config.get('coherence','REMOVE_PUNC')))
embedding_model = config.get('coherence','RETRIEVER')
embedding_model_format = config.get('coherence','RETRIEVER_FORMAT')
embedding_layer = int(config.get('coherence','RETRIEVER_EMB_LAYER'))
embedding_dim  = int(config.get('coherence','EMBEDDING_DIM'))
max_seq_len = int(config.get('coherence','MAX_SEQ_LENGTH')) 
retriever_top_k = int(config.get('coherence','RETRIEVER_TOP_K'))



def app():

    #### APP INFO #####
    with st.container():
        st.markdown("<h1 style='text-align: center;  \
                      color: black;'> NDC Comparison</h1>", 
                      unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
    with st.expander("ℹ️ - About this app", expanded=False):

        st.write(
            """     
            The *NDC Comparison* application provides easy evaluation of 
            coherence between a given policy document and a country’s (Intended)\
            Nationally Determined Contribution (INDCs/NDCs) using open-source \
            data from the German Institute of Development and Sustainability’s \
            (IDOS) [NDC Explorer](https://klimalog.idos-research.de/ndc/#NDCExplorer/worldMap?NewAndUpdatedNDC??income???catIncome).\
            """)
        st.write("")
        st.write(""" User can select a country context via the drop-down menu \
            on the left-hand side of the application. Subsequently, the user is \
            given the opportunity to manually upload another policy document \
            from the same national context or to select a pre-loaded example \
            document. Thereafter, the user can choose between two categories \
            to compare coherence between the documents: climate change adaptation \
            and climate change mitigation. Based on the selected information, \
            the application identifies relevant paragraphs in the uploaded \
            document and assigns them to the respective indicator from the NDC \
            Explorer. Currently, the NDC Explorer has 20 indicators under \
            climate change mitigation (e.g., fossil fuel production, REDD+) and \
            22 indicators under climate change adaptation (e.g., sea level rise,\
            investment needs). The assignment of the paragraph to a corresponding\
            indicator is based on vector similarities in which top 3 results
            if found are shown to the user. """)
        st.write("")
        st.write("")
        st.markdown("Some runtime metrics tested with cpu: Intel(R) Xeon(R) CPU @ 2.20GHz, memory: 13GB")
        col1,col2= st.columns(2)
        with col1:
            st.caption("OCR File processing")
            # st.markdown('<div style="text-align: center;">50 sec</div>', unsafe_allow_html=True)
            st.write("50 sec")
           
        with col2:
            st.caption("NDC comparison on 200 paragraphs(~ 35 pages)")
            # st.markdown('<div style="text-align: center;">12 sec</div>', unsafe_allow_html=True)
            st.write("140 sec")
    
    with st.sidebar:

        option = st.selectbox('Select Country', (countrynames))
        countryCode = countryList[option]
        st.markdown("---")

        genre = st.radio( "Select Category",('Climate Change Adaptation', 
                                            'Climate Change Mitigation'))
        st.markdown("---")
    
    with st.container():
        if st.button("Compare with NDC"):
            sent_cca = countrySpecificCCA(cca_sent,1,countryCode)
            sent_ccm = countrySpecificCCM(ccm_sent,1,countryCode)

            if 'filepath' in st.session_state:
                allDocuments = runSemanticPreprocessingPipeline(
                        file_path= st.session_state['filepath'],
                        file_name  = st.session_state['filename'],
                        split_by=split_by,
                        split_length= split_length,
                        split_overlap=split_overlap,
                        remove_punc= remove_punc,
                split_respect_sentence_boundary=split_respect_sentence_boundary)
                # genre = st.radio( "Select Category",('Climate Change Adaptation', 'Climate Change Mitigation'))
                if genre == 'Climate Change Adaptation':
                    sent_dict = sent_cca
                else:
                    sent_dict = sent_ccm
                sent_labels = []
                for key,sent in sent_dict.items():
                            sent_labels.append(sent)
                if len(allDocuments['documents']) > 100:
                    warning_msg = ": This might take sometime, please sit back and relax."
                else:
                    warning_msg = ""
                logging.info("starting Coherence analysis, \
                    country selected {}".format(option))
                with st.spinner("Performing Coherence Analysis for {} \
                    under {} category{}".format(option,genre,warning_msg)):
                    semanticsearch_pipeline, doc_store = semanticSearchPipeline(documents = allDocuments['documents'],
                            embedding_model= embedding_model, 
                            embedding_layer= embedding_layer,
                            embedding_model_format= embedding_model_format,
                            retriever_top_k= retriever_top_k,
                            embedding_dim=embedding_dim,
                            max_seq_len=max_seq_len, useQueryCheck=False)
                    raw_output = runSemanticPipeline(pipeline=semanticsearch_pipeline,queries=sent_labels)
                    results_df = process_semantic_output(raw_output)
                    results_df = results_df.drop(['answer','answer_offset',
                                'context_offset','context','reader_score','id'],
                                axis = 1)
                    
                    for i,key in enumerate(list(sent_dict.keys())):
                        st.subheader("Relevant paragraphs for topic: {}".format(key))
                        df = results_df[results_df['query']==sent_dict[key]].reset_index(drop=True)
                        for j in range(3):
                            st.write('Result {}.'.format(j+1))
                            st.write(df.loc[j]['content']+'\n')
                    
            else:
                st.info("🤔 No document found, please try to upload it at the sidebar!")
                logging.warning("Terminated as no document provided")