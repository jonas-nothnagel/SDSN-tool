# set path
import glob, os, sys; sys.path.append('../udfPreprocess')

#import helper
import udfPreprocess.docPreprocessing as pre
import udfPreprocess.cleaning as clean

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

import tempfile
import sqlite3

def app():

    with st.container():
        st.markdown("<h1 style='text-align: center;  \
                      color: black;'> Keyword Search</h1>", 
                      unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
            The *Keyword Search* app is an easy-to-use interface \ 
            built in Streamlit for doing keyword search in \
            policy document - developed by GIZ Data and the \
            Sustainable Development Solution Network.
            """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("### 📌 Step One: Upload document ### ")

    with st.container():
      def bm25_tokenizer(text):
            tokenized_doc = []
            for token in text.lower().split():
                token = token.strip(string.punctuation)

                if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                    tokenized_doc.append(token)
            return tokenized_doc
          
      def bm25TokenizeDoc(paraList):
          tokenized_corpus = []
          for passage in tqdm(paraList):
              if len(passage.split()) >256:
                  temp  = " ".join(passage.split()[:256])
                  tokenized_corpus.append(bm25_tokenizer(temp))
                  temp  = " ".join(passage.split()[256:])
                  tokenized_corpus.append(bm25_tokenizer(temp))
              else:
                  tokenized_corpus.append(bm25_tokenizer(passage))
                  
          return tokenized_corpus
      def search(keyword):
                ##### BM25 search (lexical search) #####
                bm25_scores = document_bm25.get_scores(bm25_tokenizer(keyword))
                top_n = np.argpartition(bm25_scores, -10)[-10:]
                bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
                bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
                
                ##### Sematic Search #####
                # Encode the query using the bi-encoder and find potentially relevant passages
                #query = "Does document contain {} issues ?".format(keyword)
                question_embedding = bi_encoder.encode(keyword, convert_to_tensor=True)
          
                hits = util.semantic_search(question_embedding, document_embeddings, top_k=top_k)
                hits = hits[0]  # Get the hits for the first query
                
                
                ##### Re-Ranking #####
                # Now, score all retrieved passages with the cross_encoder
                #cross_inp = [[query, paraList[hit['corpus_id']]] for hit in hits]
                #cross_scores = cross_encoder.predict(cross_inp)
                
                # Sort results by the cross-encoder scores
                #for idx in range(len(cross_scores)):
                  #   hits[idx]['cross-score'] = cross_scores[idx]
                  
                
                return bm25_hits, hits

      def show_results(keywordList):
        document = docx.Document()
        document.add_heading('Document name:{}'.format(file_name), 2)
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
        document.add_heading('Your Seacrhed for {}'.format(keywordList), level=1)
        for keyword in keywordList:
          
          st.write("Results for Query: {}".format(keyword))
          para = document.add_paragraph().add_run("Results for Query: {}".format(keyword))
          para.font.size = Pt(12)
          bm25_hits, hits = search(keyword)     

          st.markdown("""
                      We will provide with 2 kind of results. The 'lexical search' and the semantic search. 
                      """)  
          # In the semantic search part we provide two kind of results one with only Retriever (Bi-Encoder) and other the ReRanker (Cross Encoder)           
          st.markdown("Top few lexical search (BM25) hits")
          document.add_paragraph("Top few lexical search (BM25) hits")

          for hit in bm25_hits[0:5]:
              if hit['score'] > 0.00:   
                  st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                  document.add_paragraph("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
          
        
        
        #   st.table(bm25_hits[0:3])
          
          st.markdown("\n-------------------------\n")
          st.markdown("Top few Bi-Encoder Retrieval hits")
          document.add_paragraph("\n-------------------------\n")
          document.add_paragraph("Top few Bi-Encoder Retrieval hits")

          hits = sorted(hits, key=lambda x: x['score'], reverse=True)
          for hit in hits[0:5]:
            #  if hit['score'] > 0.45:
              st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
              document.add_paragraph("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
          #st.table(hits[0:3]
        document.save('demo.docx')
        with open("demo.docx", "rb") as file:
                     btn = st.download_button(
                     label="Download file",
                     data=file,
                     file_name="demo.docx",
                     mime="txt/docx"
                       )  


      @st.cache(allow_output_mutation=True)
      def load_sentenceTransformer(name):
          return SentenceTransformer(name)



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
              haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)

      else:
        # listing the options
        option = st.selectbox('Select the example document',
                              ('South Africa:Low Emission strategy', 
                              'Ethiopia: 10 Year Development Plan'))
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
          haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)

        if docs is not None:
          
          bi_encoder = load_sentenceTransformer('msmarco-distilbert-cos-v5') # multi-qa-MiniLM-L6-cos-v1
          bi_encoder.max_seq_length = 64     #Truncate long passages to 256 tokens
          top_k = 32

          document_embeddings = bi_encoder.encode(paraList, convert_to_tensor=True, show_progress_bar=False)
          tokenized_corpus = bm25TokenizeDoc(paraList)
          document_bm25 = BM25Okapi(tokenized_corpus)
          keywordList = None

          col1, col2 = st.columns(2)
          with col1:
            if st.button('Climate Change Keyword Search'):
              keywordList = ['extreme weather', 'floods', 'droughts']
            
             # show_results(keywordList)
          with col2:
            if st.button('Gender Keywords Search'):
              keywordList =  ['Gender', 'Women empowernment']
            
             # show_results(keywordList)
          
          keyword = st.text_input("Please enter here \
                                    what you want to search, \
                                    we will look for similar context \
                                    in the document.",
                                    value="",)
          if st.button("Find them."):
            keywordList = [keyword]
          if keywordList is not None:

              show_results(keywordList)
          



        # @st.cache(allow_output_mutation=True)
        # def load_sentenceTransformer(name):
        #     return SentenceTransformer(name)

        # bi_encoder = load_sentenceTransformer('msmarco-distilbert-cos-v5') # multi-qa-MiniLM-L6-cos-v1
        # bi_encoder.max_seq_length = 64     #Truncate long passages to 256 tokens
        # top_k = 32
        
        # #@st.cache(allow_output_mutation=True)
        # #def load_crossEncoder(name):
        #   #   return CrossEncoder(name)
        
        # # cross_encoder = load_crossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # document_embeddings = bi_encoder.encode(paraList, convert_to_tensor=True, show_progress_bar=False)

        # def bm25_tokenizer(text):
        #     tokenized_doc = []
        #     for token in text.lower().split():
        #         token = token.strip(string.punctuation)

        #         if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
        #             tokenized_doc.append(token)
        #     return tokenized_doc
            
        # def bm25TokenizeDoc(paraList):
        #     tokenized_corpus = []
        #     for passage in tqdm(paraList):
        #         if len(passage.split()) >256:
        #             temp  = " ".join(passage.split()[:256])
        #             tokenized_corpus.append(bm25_tokenizer(temp))
        #             temp  = " ".join(passage.split()[256:])
        #             tokenized_corpus.append(bm25_tokenizer(temp))
        #         else:
        #             tokenized_corpus.append(bm25_tokenizer(passage))
                    
        #     return tokenized_corpus
        
        # tokenized_corpus = bm25TokenizeDoc(paraList)
        

        # document_bm25 = BM25Okapi(tokenized_corpus)
        
        # # def search(keyword):
        # #         ##### BM25 search (lexical search) #####
        # #         bm25_scores = document_bm25.get_scores(bm25_tokenizer(keyword))
        #         top_n = np.argpartition(bm25_scores, -10)[-10:]
        #         bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        #         bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
                
        #         ##### Sematic Search #####
        #         # Encode the query using the bi-encoder and find potentially relevant passages
        #         #query = "Does document contain {} issues ?".format(keyword)
        #         question_embedding = bi_encoder.encode(keyword, convert_to_tensor=True)
          
        #         hits = util.semantic_search(question_embedding, document_embeddings, top_k=top_k)
        #         hits = hits[0]  # Get the hits for the first query
                
                
        #         ##### Re-Ranking #####
        #         # Now, score all retrieved passages with the cross_encoder
        #         #cross_inp = [[query, paraList[hit['corpus_id']]] for hit in hits]
        #         #cross_scores = cross_encoder.predict(cross_inp)
                
        #         # Sort results by the cross-encoder scores
        #         #for idx in range(len(cross_scores)):
        #           #   hits[idx]['cross-score'] = cross_scores[idx]
                  
                
        #         return bm25_hits, hits

        # def show_results(keywordList):
        #   for keyword in keywordList:
        #     bm25_hits, hits = search(keyword)     

        #     st.markdown("""
        #                 We will provide with 2 kind of results. The 'lexical search' and the semantic search. 
        #                 """)  
        #     # In the semantic search part we provide two kind of results one with only Retriever (Bi-Encoder) and other the ReRanker (Cross Encoder)           
        #     st.markdown("Top few lexical search (BM25) hits")
        #     for hit in bm25_hits[0:5]:
        #         if hit['score'] > 0.00:   
        #             st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
            
            
          
          
          
        #   #   st.table(bm25_hits[0:3])
            
        #     st.markdown("\n-------------------------\n")
        #     st.markdown("Top few Bi-Encoder Retrieval hits")
            
        #     hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        #     for hit in hits[0:5]:
        #       #  if hit['score'] > 0.45:
        #         st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
        #     #st.table(hits[0:3]


        # # if docs is not None:
        # #     col1, col2 = st.columns(2)
        # #     with col1:
        # #       if st.button('Gender Keywords Search'):
        # #         keywordList =  ['Gender Equality', 'Women empowernment']
        # #         show_results(keywordList)
        # #     with col2:
        # #       if st.button('Climate Change Keyword Search'):
        # #         keywordList = ['extreme weather', 'floods', 'droughts']
        # #         show_results(keywordList)
            
        # #     keyword = st.text_input("Please enter here \
        # #                              what you want to search, \
        # #                              we will look for similar context \
        # #                              in the document.",
        # #                              value="",)
        # #     if st.button("Find them."):
        # #       show_results([keyword])

  
            # choice1 = st.radio(label = 'Keyword Search',
            #               help = 'Search  \
            #               or else you can try a example document', 
            #               options = ('Enter your own Query', 'Try Example'), 
            #               horizontal = True)
            
            # if choice1 == 'Enter your own Query':
            #   keyword = st.text_input("Please enter here \
            #                         what you want to search, \
            #                         we will look for similar context \
            #                         in the document.",
            #                         value="",)
            # else:
            #   option1 = st.selectbox('Select the Predefined word cluster',
            #                     ('Gender:[Gender Equality, Women empowernment]', 
            #                     'Climate change:[extreme weather, floods, droughts]',
            #                     ))
            #   if option1 == 'Gender:[Gender Equality, Women empowernment]':
            #     keywordList = ['Gender Equality', 'Women empowernment']
            #   else:
            #     keywordList = ['extreme weather', 'floods', 'droughts']

            # option1 = st.selectbox('Select the Predefined word cluster',
            #                     ('Gender:[Gender Equality, Women empowernment]', 
            #                     'Climate change:[extreme weather, floods, droughts]',
            # #                     'Enter your Own Keyword Query'))
            # if option1 == 'Enter your Own Keyword Query':
            #   keyword = st.text_input("Please enter here \
            #                         what you want to search, \
            #                         we will look for similar context \
            #                         in the document.",
            #                         value="",)
            # elif option1 == 'Gender:[Gender Equality, Women empowernment]':
            #   keywordList = ['Gender Equality', 'Women empowernment']
            # elif option1 == 'Climate change:[extreme weather, floods, droughts]':
            #   keywordList = ['extreme weather', 'floods', 'droughts']


            # st.markdown("### 📌 Step Two: Search Keyword in Document ### ")             
            
                                      
            # @st.cache(allow_output_mutation=True)
            # def load_sentenceTransformer(name):
            #     return SentenceTransformer(name)

            # bi_encoder = load_sentenceTransformer('msmarco-distilbert-cos-v5') # multi-qa-MiniLM-L6-cos-v1
            # bi_encoder.max_seq_length = 64     #Truncate long passages to 256 tokens
            # top_k = 32
            
            # #@st.cache(allow_output_mutation=True)
            # #def load_crossEncoder(name):
            #   #   return CrossEncoder(name)
            
            # # cross_encoder = load_crossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            # document_embeddings = bi_encoder.encode(paraList, convert_to_tensor=True, show_progress_bar=False)

            # def bm25_tokenizer(text):
            #     tokenized_doc = []
            #     for token in text.lower().split():
            #         token = token.strip(string.punctuation)

            #         if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            #             tokenized_doc.append(token)
            #     return tokenized_doc
                
            # def bm25TokenizeDoc(paraList):
            #     tokenized_corpus = []
            #     for passage in tqdm(paraList):
            #         if len(passage.split()) >256:
            #             temp  = " ".join(passage.split()[:256])
            #             tokenized_corpus.append(bm25_tokenizer(temp))
            #             temp  = " ".join(passage.split()[256:])
            #             tokenized_corpus.append(bm25_tokenizer(temp))
            #         else:
            #             tokenized_corpus.append(bm25_tokenizer(passage))
                        
            #     return tokenized_corpus
            
            # tokenized_corpus = bm25TokenizeDoc(paraList)
            

            # document_bm25 = BM25Okapi(tokenized_corpus)
            
            
            # def search(keyword):
            #     ##### BM25 search (lexical search) #####
            #     bm25_scores = document_bm25.get_scores(bm25_tokenizer(keyword))
            #     top_n = np.argpartition(bm25_scores, -10)[-10:]
            #     bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
            #     bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
                
            #     ##### Sematic Search #####
            #     # Encode the query using the bi-encoder and find potentially relevant passages
            #     #query = "Does document contain {} issues ?".format(keyword)
            #     question_embedding = bi_encoder.encode(keyword, convert_to_tensor=True)
          
            #     hits = util.semantic_search(question_embedding, document_embeddings, top_k=top_k)
            #     hits = hits[0]  # Get the hits for the first query
                
                
            #     ##### Re-Ranking #####
            #     # Now, score all retrieved passages with the cross_encoder
            #     #cross_inp = [[query, paraList[hit['corpus_id']]] for hit in hits]
            #     #cross_scores = cross_encoder.predict(cross_inp)
                
            #     # Sort results by the cross-encoder scores
            #     #for idx in range(len(cross_scores)):
            #       #   hits[idx]['cross-score'] = cross_scores[idx]
                  
                
            #     return bm25_hits, hits

            # def show_results(keywordList):
            #   for keyword in keywordList:
            #     bm25_hits, hits = search(keyword)     

            #     st.markdown("""
            #                 We will provide with 2 kind of results. The 'lexical search' and the semantic search. 
            #                 """)  
            #     # In the semantic search part we provide two kind of results one with only Retriever (Bi-Encoder) and other the ReRanker (Cross Encoder)           
            #     st.markdown("Top few lexical search (BM25) hits")
            #     for hit in bm25_hits[0:5]:
            #         if hit['score'] > 0.00:   
            #             st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                
                
              
              
              
            #   #   st.table(bm25_hits[0:3])
                
            #     st.markdown("\n-------------------------\n")
            #     st.markdown("Top few Bi-Encoder Retrieval hits")
                
            #     hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            #     for hit in hits[0:5]:
            #       #  if hit['score'] > 0.45:
            #         st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
            #     #st.table(hits[0:3]
        

          

            # # if st.button("Find them."):
            # #     bm25_hits, hits = search(keyword)     

            # #     st.markdown("""
            # #                 We will provide with 2 kind of results. The 'lexical search' and the semantic search. 
            # #                 """)  
            # #     # In the semantic search part we provide two kind of results one with only Retriever (Bi-Encoder) and other the ReRanker (Cross Encoder)           
            # #     st.markdown("Top few lexical search (BM25) hits")
            # #     for hit in bm25_hits[0:5]:
            # #         if hit['score'] > 0.00:   
            # #             st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
                
                
              
              
              
            # #   #   st.table(bm25_hits[0:3])
                
            # #     st.markdown("\n-------------------------\n")
            # #     st.markdown("Top few Bi-Encoder Retrieval hits")
                
            # #     hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            # #     for hit in hits[0:5]:
            # #       #  if hit['score'] > 0.45:
            # #         st.write("\t Score: {:.3f}:  \t{}".format(hit['score'], paraList[hit['corpus_id']].replace("\n", " ")))
            # #     #st.table(hits[0:3]
        

          