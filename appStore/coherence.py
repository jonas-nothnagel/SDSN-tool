# set path
import glob, os, sys; 
sys.path.append('../utils')

import streamlit as st
import ast

# Reading data and Declaring necessary variables
with open('ndcs/countryList.txt') as dfile:
        countryList = dfile.read()
countryList = ast.literal_eval(countryList)
countrynames = list(countryList.keys())
    
with open('ndcs/cca.txt', encoding='utf-8', errors='ignore') as dfile:
            cca_sent = dfile.read()
cca_sent = ast.literal_eval(cca_sent)
            
with open('ndcs/ccm.txt', encoding='utf-8', errors='ignore') as dfile:
    ccm_sent = dfile.read()
ccm_sent = ast.literal_eval(ccm_sent)

def app():

    #### APP INFO #####
    with st.container():
        st.markdown("<h1 style='text-align: center;  \
                      color: black;'> Check NDC Coherence</h1>", 
                      unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
    with st.expander("ℹ️ - About this app", expanded=False):

        st.write(
            """     
            The *Check NDC Coherence* application provides easy evaluation of 
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
            indicator is based on vector similarities in which only paragraphs \
            with similarity above 0.55  to the indicators are considered. """)
    
    option = st.sidebar.selectbox('Select Country', (countrynames))
    countryCode = countryList[option]
    