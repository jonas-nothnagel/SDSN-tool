# set path
import glob, os, sys; sys.path.append('../udfPreprocess')

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

import tempfile
import sqlite3

def app():

    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'> SDSN x GIZ Policy Action Tracking v0.1</h1>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
            The *Analyse Policy Document* app is an easy-to-use interface built in Streamlit for analyzing policy documents - developed by GIZ Data and the Sustainable Development Solution Network. \n
                1. Keyword heatmap \n
                2. SDG Classification for the paragraphs/texts in the document
            """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("##  📌 Step One: Upload document ")
    
    with st.container():


      docs = None
        # asking user for either upload or select existing doc
      choice = st.radio(label = 'Select the Document',
                        help = 'You can upload the document \
                        or else you can try a example document', 
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
              docs_processed, df, all_text, par_list = clean.preprocessingForSDG(docs)
              #haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)

      else:
        # listing the options
        option = st.selectbox('Select the example document',
                              ('Ethiopia: 10 Year Development Plan',
                              'South Africa:Low Emission strategy'))
        if option is 'South Africa:Low Emission strategy':
          file_name = file_path  = 'sample/South Africa_s Low Emission Development Strategy.txt'
          st.write("Selected document:", file_name.split('/')[1])
          # with open('sample/South Africa_s Low Emission Development Strategy.txt') as dfile:
          # file = open('sample/South Africa_s Low Emission Development Strategy.txt', 'wb')
        else:
          # with open('sample/Ethiopia_s_2021_10 Year Development Plan.txt') as dfile:
          file_name = file_path =  'sample/Ethiopia_s_2021_10 Year Development Plan.txt'
          st.write("Selected document:", file_name.split('/')[1])
        
        if option is not None:
          docs = pre.load_document(file_path,file_name)
          # haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)
          docs_processed, df, all_text, par_list = clean.preprocessingForSDG(docs)

        
        
      if docs is not None:

                @st.cache(allow_output_mutation=True)
                def load_keyBert():
                    return KeyBERT()

                kw_model = load_keyBert()

                keywords = kw_model.extract_keywords(
                all_text,
                keyphrase_ngram_range=(1, 3),
                use_mmr=True,
                stop_words="english",
                top_n=10,
                diversity=0.7,
                )

                st.markdown("## 🎈 What is my document about?")
            
                df = (
                    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
                    .sort_values(by="Relevancy", ascending=False)
                    .reset_index(drop=True)
                )
                df1 = (
                    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
                    .sort_values(by="Relevancy", ascending=False)
                    .reset_index(drop=True)
                )
                df.index += 1

                # Add styling
                cmGreen = sns.light_palette("green", as_cmap=True)
                cmRed = sns.light_palette("red", as_cmap=True)
                df = df.style.background_gradient(
                    cmap=cmGreen,
                    subset=[
                        "Relevancy",
                    ],
                )
                c1, c2, c3 = st.columns([1, 3, 1])

                format_dictionary = {
                    "Relevancy": "{:.1%}",
                }

                df = df.format(format_dictionary)

                with c2:
                    st.table(df) 

                ######## SDG classiciation
                # @st.cache(allow_output_mutation=True)
                # def load_sdgClassifier():
                #     classifier = pipeline("text-classification", model= "../models/osdg_sdg/")

                #     return classifier
                
                # load from disc (github repo) for performance boost
                @st.cache(allow_output_mutation=True)
                def load_sdgClassifier():
                    classifier = pipeline("text-classification", model= "jonas/sdg_classifier_osdg")

                    return classifier

                classifier = load_sdgClassifier()

                # # not needed, par list comes from pre_processing function already

                # word_list = all_text.split()
                # len_word_list = len(word_list)
                # par_list = []
                # par_len = 130
                # for i in range(0,len_word_list // par_len):
                #     string_part = ' '.join(word_list[i*par_len:(i+1)*par_len])
                #     par_list.append(string_part)
                    
                labels = classifier(par_list)
                labels_= [(l['label'],l['score']) for l in labels]
                df2 = DataFrame(labels_, columns=["SDG", "Relevancy"])
                df2['text'] = par_list      
                df2 = df2.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
                df2.index += 1
                df2 =df2[df2['Relevancy']>.85]
                x = df2['SDG'].value_counts()
                df3 = df2.copy()

                plt.rcParams['font.size'] = 25
                colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
                # plot
                fig, ax = plt.subplots()
                ax.pie(x, colors=colors, radius=2, center=(4, 4),
                    wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=False,labels =list(x.index))
                fig.savefig('temp.png', bbox_inches='tight',dpi= 100)
                st.markdown("## 🎈 Anything related to SDGs?")

                c4, c5, c6 = st.columns([2, 2, 2])

                # Add styling
                cmGreen = sns.light_palette("green", as_cmap=True)
                cmRed = sns.light_palette("red", as_cmap=True)
                df2 = df2.style.background_gradient(
                    cmap=cmGreen,
                    subset=[
                        "Relevancy",
                    ],
                )

                format_dictionary = {
                    "Relevancy": "{:.1%}",
                }

                df2 = df2.format(format_dictionary)

                with c5:
                    st.pyplot(fig)
                    
                c7, c8, c9 = st.columns([1, 10, 1])
                with c8:
                    st.table(df2) 

                document = docx.Document()
                document.add_heading('Document name:{}'.format(file_name), 2)
                # Choosing the top most section of the page
                section = document.sections[0]
 
                # Calling the footer
                footer = section.footer
                
                # Calling the paragraph already present in
                # the footer section
                footer_para = footer.paragraphs[0]
                
                font_styles = document.styles
                font_charstyle = font_styles.add_style('CommentsStyle', WD_STYLE_TYPE.CHARACTER)
                font_object = font_charstyle.font
                font_object.size = Pt(7)
                # Adding the centered zoned footer
                footer_para.add_run('''\tPowered by GIZ Data and the Sustainable Development Solution Network hosted at Hugging-Face spaces: https://huggingface.co/spaces/ppsingh/streamlit_dev''', style='CommentsStyle')
                
                #footer_para.text = "\tPowered by GIZ Data and the Sustainable Development Solution Network\
                 #                     hosted at Hugging-Face spaces: https://huggingface.co/spaces/ppsingh/streamlit_dev"
                #footer_para.font.size = docx.shared.Pt(6)                     
                
                document.add_heading('What is the document about', level=1)
                t = document.add_table(df1.shape[0]+1, df1.shape[1])
                

                # add the header rows.
                for j in range(df1.shape[-1]):
                    t.cell(0,j).text = df1.columns[j]
                    

                # add the rest of the data frame
                for i in range(df1.shape[0]):
                    for j in range(df1.shape[-1]):
                        t.cell(i+1,j).text = str(df1.values[i,j])
                        
                        
                        
                document.add_heading('Anything Related to SDG', level=1)        
                document.add_picture('temp.png', width=Inches(3), height=Inches(3)) 
                t = document.add_table(df3.shape[0]+1, df3.shape[1])
                
                widths = [Inches(0.4), Inches(0.4), Inches(4.5)]
                # add the header rows.
                for j in range(df3.shape[-1]):
                    t.cell(0,j).text = df3.columns[j]
                    t.cell(0,j).width = widths[j]

                # add the rest of the data frame
                for i in range(df3.shape[0]):
                    for j in range(df3.shape[-1]):
                        t.cell(i+1,j).width = widths[j]
                        t.cell(i+1,j).text = str(df3.values[i,j])
                        
                              
                document.save('demo.docx')
                
                #with open('summary.txt', 'w') as f:
                 #   f.write(df1.to_string())
                 #   f.write(fig)
                    #f.write(df2)
                    # f.write(df3.to_string())
                    
                with open("demo.docx", "rb") as file:
                     btn = st.download_button(
                     label="Download file",
                     data=file,
                     file_name="demo.docx",
                     mime="txt/docx"
                       )    
                #with document st.download_button(
                 #  label="Download data as docx",
                  # data=document,
                   #file_name='test.docx',
                   #mime='text/docx',
                    # )    
                
                
