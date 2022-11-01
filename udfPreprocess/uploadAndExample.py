import streamlit as st
import tempfile

def add_upload(choice):
    """
    Provdies the user with choice to either 'Upload Document' or 'Try Example'.
    Based on user choice runs streamlit processes and save the path and name of
    the 'file' to streamlit session_state which then can be fetched later.

    """

    
    if choice == 'Upload Document':
        uploaded_file = st.sidebar.file_uploader('Upload the File',
                            type=['pdf', 'docx', 'txt'])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(mode="wb") as temp:
                bytes_data = uploaded_file.getvalue()
                temp.write(bytes_data)
                st.session_state['filename'] = uploaded_file.name
                file_name =  uploaded_file.name
                file_path = temp.name
                st.session_state['filename'] = file_name
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
        else:
            file_name = file_path =  'sample/Ethiopia_s_2021_10 Year Development Plan.txt'
            st.session_state['filename'] = file_name
            st.session_state['filepath'] = file_path