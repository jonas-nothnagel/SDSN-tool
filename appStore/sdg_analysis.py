# set path
import glob, os, sys; 
sys.path.append('../udfPreprocess')

#import helper
import udfPreprocess.docPreprocessing as pre
import udfPreprocess.cleaning as clean

#import needed libraries
import seaborn as sns
from pandas import DataFrame
from keybert import KeyBERT
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import docx
from docx.shared import Inches
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
from udfPreprocess.sdg_classifier import sdg_classification
from udfPreprocess.sdg_classifier import runSDGPreprocessingPipeline
import configparser
import tempfile
import sqlite3
import logging
logger = logging.getLogger(__name__)



# @st.cache(allow_output_mutation=True)
# def load_keyBert():
#     return KeyBERT()

# @st.cache(allow_output_mutation=True)
# def load_sdgClassifier():
#     classifier = pipeline("text-classification", model= "jonas/sdg_classifier_osdg")
#     return classifier



def app():

    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'> SDSN x GIZ Policy Action Tracking v0.1</h1>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=False):

        st.write(
            """     
            The *Analyse Policy Document* app is an easy-to-use interface built in Streamlit for analyzing policy documents with respect to SDG Classification for the paragraphs/texts in the document - developed by GIZ Data and the Sustainable Development Solution Network. \n
            """)
        st.markdown("")


    with st.container():
        

            
        if 'filepath' in st.session_state:
            paraList = runSDGPreprocessingPipeline()
            with st.spinner("Running SDG"):

                df, x = sdg_classification(paraList)


                # classifier = load_sdgClassifier()

                # labels = classifier(par_list)
                # labels_= [(l['label'],l['score']) for l in labels]
                # df2 = DataFrame(labels_, columns=["SDG", "Relevancy"])
                # df2['text'] = par_list      
                # df2 = df2.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
                # df2.index += 1
                # df2 =df2[df2['Relevancy']>.85]
                # x = df2['SDG'].value_counts()
                # df3 = df2.copy()

                plt.rcParams['font.size'] = 25
                colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
                # plot
                fig, ax = plt.subplots()
                ax.pie(x, colors=colors, radius=2, center=(4, 4),
                    wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=False,labels =list(x.index))
                # fig.savefig('temp.png', bbox_inches='tight',dpi= 100)
                st.markdown("#### Anything related to SDGs? ####")

                # st.markdown("#### 🎈 Anything related to SDGs? ####")

                c4, c5, c6 = st.columns([2, 2, 2])

                # Add styling
                cmGreen = sns.light_palette("green", as_cmap=True)
                cmRed = sns.light_palette("red", as_cmap=True)
                # df2 = df2.style.background_gradient(
                #     cmap=cmGreen,
                #     subset=[
                #         "Relevancy",
                #     ],
                # )

                # format_dictionary = {
                #     "Relevancy": "{:.1%}",
                # }

                # df2 = df2.format(format_dictionary)

                with c5:
                    st.pyplot(fig)
                    
                c7, c8, c9 = st.columns([1, 10, 1])
                with c8:
                    st.table(df)


#     1. Keyword heatmap \n
 #               2. SDG Classification for the paragraphs/texts in the document
 #       
    
    # with st.container():
    #     if 'docs' in st.session_state:
    #         docs = st.session_state['docs']
    #         docs_processed, df, all_text, par_list = clean.preprocessingForSDG(docs)
    #         # paraList = st.session_state['paraList']
    #         logging.info("keybert")
    #         with st.spinner("Running Key bert"):

    #             kw_model = load_keyBert()

    #             keywords = kw_model.extract_keywords(
    #             all_text,
    #             keyphrase_ngram_range=(1, 3),
    #             use_mmr=True,
    #             stop_words="english",
    #             top_n=10,
    #             diversity=0.7,
    #             )

    #             st.markdown("## 🎈 What is my document about?")
            
    #             df = (
    #                 DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    #                 .sort_values(by="Relevancy", ascending=False)
    #                 .reset_index(drop=True)
    #             )
    #             df1 = (
    #                 DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    #                 .sort_values(by="Relevancy", ascending=False)
    #                 .reset_index(drop=True)
    #             )
    #             df.index += 1

    #             # Add styling
    #             cmGreen = sns.light_palette("green", as_cmap=True)
    #             cmRed = sns.light_palette("red", as_cmap=True)
    #             df = df.style.background_gradient(
    #                 cmap=cmGreen,
    #                 subset=[
    #                     "Relevancy",
    #                 ],
    #             )

    #             c1, c2, c3 = st.columns([1, 3, 1])

    #             format_dictionary = {
    #                 "Relevancy": "{:.1%}",
    #             }

    #             df = df.format(format_dictionary)

    #             with c2:
    #  
    #               st.table(df)