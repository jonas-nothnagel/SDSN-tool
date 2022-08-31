# set path
import glob, os, sys; sys.path.append('../udfPreprocess')

#import helper
import udfPreprocess.docPreprocessing as pre
import udfPreprocess.cleaning as clean

#import needed libraries
import seaborn as sns
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.metrics.pairwise import cosine_similarity
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
import urllib.request
import ast
import tempfile
import sqlite3
import json
import urllib.request
import ast
def app():
    # Sidebar
    st.sidebar.title('Check Coherence')
    st.sidebar.write(' ')
    with open('ndcs/countryList.txt') as dfile:
        countryList = dfile.read()

    countryList = ast.literal_eval(countryList)
    countrynames = list(countryList.keys())
    
    option = st.sidebar.selectbox('Select Country', (countrynames))
    countryCode = countryList[option]


    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'> Check Coherence of Policy Document with NDCs</h1>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
            The *Check Coherence* app is an easy-to-use interface built in Streamlit for doing analysis of policy document and finding the coherence between NDCs/New-Updated NDCs- developed by GIZ Data and the Sustainable Development Solution Network.
            """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("##  📌 Step One: Upload document of the country selected ")
    
    with st.container():
            docs = None
            # asking user for either upload or select existing doc
            choice = st.radio(label = 'Select the Document',
                              help = 'You can upload the document \
                              or else you can try a example document.', 
                              options = ('Upload Document', 'Try Example'), 
                              horizontal = True)

            if choice == 'Upload Document':
              uploaded_file = st.file_uploader('Upload the File', type=['pdf', 'docx', 'txt'])
              if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(mode="wb") as temp:
                    bytes_data = uploaded_file.getvalue()
                    temp.write(bytes_data)

                    st.write("Uploaded Filename: ", uploaded_file.name)
                    file_name =  uploaded_file.name
                    file_path = temp.name
                    docs = pre.load_document(file_path, file_name)
                    haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)

            else:
              # listing the options
              option = st.selectbox('Select the example document',
                                    ('South Africa:Low Emission strategy', 
                                    'Ethiopia: 10 Year Development Plan'))
              if option is 'South Africa:Low Emission strategy':
                file_name = file_path  = 'sample/South Africa_s Low Emission Development Strategy.txt'
                countryCode = countryList['South Africa']
                st.write("Selected document:", file_name.split('/')[1])
                # with open('sample/South Africa_s Low Emission Development Strategy.txt') as dfile:
                # file = open('sample/South Africa_s Low Emission Development Strategy.txt', 'wb')
              else:
                # with open('sample/Ethiopia_s_2021_10 Year Development Plan.txt') as dfile:
                file_name = file_path =  'sample/Ethiopia_s_2021_10 Year Development Plan.txt'
                countryCode = countryList['Ethiopia']
                st.write("Selected document:", file_name.split('/')[1])
              
              if option is not None:
                docs = pre.load_document(file_path,file_name)
                haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)

            with open('ndcs/cca.txt', encoding='utf-8', errors='ignore') as dfile:
                cca_sent = dfile.read()

            cca_sent = ast.literal_eval(cca_sent)
            
            with open('ndcs/ccm.txt', encoding='utf-8', errors='ignore') as dfile:
                ccm_sent = dfile.read()

            ccm_sent = ast.literal_eval(ccm_sent)
            
            with open('ndcs/countryList.txt') as dfile:
                countryList = dfile.read()

            countryList = ast.literal_eval(countryList)
            
            def get_document(countryCode: str):
                link = "https://klimalog.die-gdi.de/ndc/open-data/dataset.json"  
                with urllib.request.urlopen(link) as urlfile:
                    data =  json.loads(urlfile.read())
                categoriesData = {}
                categoriesData['categories']= data['categories']
                categoriesData['subcategories']= data['subcategories']
                keys_sub = categoriesData['subcategories'].keys()
                documentType= 'NDCs'
                if documentType in data.keys():
                    if countryCode in data[documentType].keys():
                        get_dict = {}
                        for key, value in data[documentType][countryCode].items():
                            if key not in ['country_name','region_id', 'region_name']:
                                get_dict[key] = value['classification']
                            else:
                                get_dict[key] = value
                    else:
                        return None
                else:
                    return None

                country = {}
                for key in categoriesData['categories']:
                    country[key]= {}
                for key,value in categoriesData['subcategories'].items():
                    country[value['category']][key] = get_dict[key]
                
                return country
        
        #   country_ndc = get_document('NDCs', countryList[option])
            
            def countrySpecificCCA(cca_sent, threshold, countryCode):
                temp = {}
                doc = get_document(countryCode)
                for key,value in cca_sent.items():
                    id_ = doc['climate change adaptation'][key]['id']
                    if id_ >threshold:
                        temp[key] = value['id'][id_]
                return temp
            
                
            def countrySpecificCCM(ccm_sent, threshold, countryCode):
                temp = {}
                doc = get_document(countryCode)
                for key,value in ccm_sent.items():
                    id_ = doc['climate change mitigation'][key]['id']
                    if id_ >threshold:
                        temp[key] = value['id'][id_]
                
                return temp

        
        
            if docs is not None:
                    sent_cca = countrySpecificCCA(cca_sent,1,countryCode)
                    sent_ccm = countrySpecificCCM(ccm_sent,1,countryCode)
                    #st.write(sent_ccm)
                    @st.cache(allow_output_mutation=True)
                    def load_sentenceTransformer(name):
                        return SentenceTransformer(name)
                    model = load_sentenceTransformer('all-MiniLM-L6-v2')
          
                    document_embeddings = model.encode(paraList, show_progress_bar=True)
                
                    genre = st.radio( "Select Category",('Climate Change Adaptation', 'Climate Change Mitigation'))
                    if genre == 'Climate Change Adaptation':
                        sent_dict = sent_cca
                        sent_labels = []
                        for key,sent in sent_dict.items():
                            sent_labels.append(sent)
                        label_embeddings = model.encode(sent_labels, show_progress_bar=True)
                        similarity_high_threshold = 0.55
                        similarity_matrix = cosine_similarity(label_embeddings, document_embeddings)
                        label_indices, paragraph_indices = np.where(similarity_matrix>similarity_high_threshold)

                        positive_indices = list(zip(label_indices.tolist(), paragraph_indices.tolist()))
                        
                        
                    else:
                        sent_dict = sent_ccm
                        sent_labels = []
                        for key,sent in sent_dict.items():
                            sent_labels.append(sent)
                        label_embeddings = model.encode(sent_labels, show_progress_bar=True)
                        similarity_high_threshold = 0.55
                        similarity_matrix = cosine_similarity(label_embeddings, document_embeddings)
                        label_indices, paragraph_indices = np.where(similarity_matrix>similarity_high_threshold)

                        positive_indices = list(zip(label_indices.tolist(), paragraph_indices.tolist()))
                        
            
                #    sent_labels = []
                #   for key,sent in sent_dict.items():
                  #      sent_labels.append(sent)
                    
            
                  # label_embeddings = model.encode(sent_labels, show_progress_bar=True)
            
                    #similarity_high_threshold = 0.55
                  # similarity_matrix = cosine_similarity(label_embeddings, document_embeddings)
                    #label_indices, paragraph_indices = np.where(similarity_matrix>similarity_high_threshold)

                    #positive_indices = list(zip(label_indices.tolist(), paragraph_indices.tolist()))
            
                    for _label_idx, _paragraph_idx in positive_indices:
                        st.write("This paragraph: \n")
                        st.write(paraList[_paragraph_idx])
                        st.write(f"Is relevant to: \n {list(sent_dict.keys())[_label_idx]}")
                        st.write('-'*10)
            
