import streamlit as st
import tempfile
import udfPreprocess.docPreprocessing as pre
import udfPreprocess.cleaning as clean

def add_upload(choice):

    
    if choice == 'Upload Document':
          uploaded_file = st.sidebar.file_uploader('Upload the File', type=['pdf', 'docx', 'txt'])
          if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(mode="wb") as temp:
                bytes_data = uploaded_file.getvalue()
                temp.write(bytes_data)
                st.session_state['filename'] = uploaded_file.name
                # st.write("Uploaded Filename: ", uploaded_file.name)
                file_name =  uploaded_file.name
                file_path = temp.name
                # docs = pre.load_document(file_path, file_name)
                # haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)
                st.session_state['filename'] = file_name
                # st.session_state['paraList'] = paraList
                st.session_state['filepath'] = file_path

                

    else:
          # listing the options
          option = st.sidebar.selectbox('Select the example document',
                                ('South Africa:Low Emission strategy', 
                                'Ethiopia: 10 Year Development Plan'))
          if option is 'South Africa:Low Emission strategy':
            file_name = file_path  = 'sample/South Africa_s Low Emission Development Strategy.txt'
            st.session_state['filename'] = file_name
            st.sesion_state['filepath'] = file_path
            # st.write("Selected document:", file_name.split('/')[1])
            # with open('sample/South Africa_s Low Emission Development Strategy.txt') as dfile:
            # file = open('sample/South Africa_s Low Emission Development Strategy.txt', 'wb')
          else:
            # with open('sample/Ethiopia_s_2021_10 Year Development Plan.txt') as dfile:
            file_name = file_path =  'sample/Ethiopia_s_2021_10 Year Development Plan.txt'
            st.session_state['filename'] = file_name
            st.session_state['filepath'] = file_path
            # st.write("Selected document:", file_name.split('/')[1])
          
          # if option is not None:
          #   docs = pre.load_document(file_path,file_name)
          #   haystackDoc, dataframeDoc, textData, paraList = clean.preprocessing(docs)
          #   st.session_state['docs'] = docs
          #   st.session_state['paraList'] = paraList
          
    