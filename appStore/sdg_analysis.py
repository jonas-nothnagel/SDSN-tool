# set path
import glob, os, sys; 
sys.path.append('../utils')

#import needed libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.shared import ColumnsAutoSizeMode
from utils.sdg_classifier import sdg_classification
from utils.sdg_classifier import runSDGPreprocessingPipeline, load_sdgClassifier
from utils.keyword_extraction import textrank
import logging
logger = logging.getLogger(__name__)
from utils.checkconfig import getconfig


# Declare all the necessary variables
config = getconfig('paramconfig.cfg')
model_name = config.get('sdg','MODEL')
split_by = config.get('sdg','SPLIT_BY')
split_length = int(config.get('sdg','SPLIT_LENGTH'))
split_overlap = int(config.get('sdg','SPLIT_OVERLAP'))
remove_punc = bool(int(config.get('sdg','REMOVE_PUNC')))
split_respect_sentence_boundary = bool(int(config.get('sdg','RESPECT_SENTENCE_BOUNDARY')))
threshold = float(config.get('sdg','THRESHOLD'))
top_n = int(config.get('sdg','TOP_KEY'))


def app():

    #### APP INFO #####
    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'> SDG Classification and Keyphrase Extraction </h1>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')

    with st.expander("ℹ️ - About this app", expanded=False):

        st.write(
            """     
            The *SDG Analysis* app is an easy-to-use interface built \
                in Streamlit for analyzing policy documents with respect to SDG \
                 Classification for the paragraphs/texts in the document and \
                extracting the keyphrase per SDG label - developed by GIZ Data \
                 and the Sustainable Development Solution Network. \n
            """)
        st.write("""**Document Processing:** The Uploaded/Selected document is \
            automatically cleaned and split into paragraphs with a maximum \
            length of 120 words using a Haystack preprocessing pipeline. The \
            length of 120 is an empirical value which should reflect the length \
            of a “context” and should limit the paragraph length deviation. \
            However, since we want to respect the sentence boundary the limit \
            can breach and hence this limit of 120 is tentative.  \n
            """)
        st.write("""**SDG cLassification:** The application assigns paragraphs \
            to 15 of the 17 United Nations Sustainable Development Goals (SDGs).\
            SDG 16 “Peace, Justice and Strong Institutions” and SDG 17 \
            “Partnerships for the Goals” are excluded from the analysis due to \
            their broad nature which could potentially inflate the results. \
            Each paragraph is assigned to one SDG only. Again, the results are \
            displayed in a summary table including the number of the SDG, a \
            relevancy score highlighted through a green color shading, and the \
            respective text of the analyzed paragraph. Additionally, a pie \
            chart with a blue color shading is displayed which illustrates the \
            three most prominent SDGs in the document. The SDG classification \
            uses open-source training [data](https://zenodo.org/record/5550238#.Y25ICHbMJPY) \
            from [OSDG.ai](https://osdg.ai/) which is a global \
            partnerships and growing community of researchers and institutions \
            interested in the classification of research according to the \
            Sustainable Development Goals. The summary table only displays \
            paragraphs with a calculated relevancy score above 85%.  \n""")

        st.write("""**Keyphrase Extraction:** The application extracts 15 \
            keyphrases from the document, for each SDG label and displays the \
            results in a summary table. The keyphrases are extracted using \
            using [Textrank](https://github.com/summanlp/textrank)\
            which is an easy-to-use computational less expensive \
            model leveraging combination of TFIDF and Graph networks.
            """)
        st.write("")
        st.write("")
        st.markdown("Some runtime metrics tested with cpu: Intel(R) Xeon(R) CPU @ 2.20GHz, memory: 13GB")
        col1,col2,col3,col4 = st.columns([2,2,4,4])
        with col1:
            st.caption("Loading Time Classifier")
            # st.markdown('<div style="text-align: center;">12 sec</div>', unsafe_allow_html=True)
            st.write("12 sec")
        with col2:
            st.caption("OCR File processing")
            # st.markdown('<div style="text-align: center;">50 sec</div>', unsafe_allow_html=True)
            st.write("50 sec")
        with col3:
            st.caption("SDG Classification of 200 paragraphs(~ 35 pages)")
            # st.markdown('<div style="text-align: center;">120 sec</div>', unsafe_allow_html=True)
            st.write("120 sec")
        with col4:
            st.caption("Keyword extraction for 200 paragraphs(~ 35 pages)")
            # st.markdown('<div style="text-align: center;">3 sec</div>', unsafe_allow_html=True)
            st.write("3 sec")

        

    
    ### Main app code ###
    with st.container():
        if st.button("RUN SDG Analysis"):
                   
            if 'filepath' in st.session_state:
                file_name = st.session_state['filename']
                file_path = st.session_state['filepath']
                classifier = load_sdgClassifier(classifier_name=model_name)
                st.session_state['sdg_classifier'] = classifier
                all_documents = runSDGPreprocessingPipeline(file_name= file_name,
                                        file_path= file_path, split_by= split_by,
                                        split_length= split_length,
                split_respect_sentence_boundary= split_respect_sentence_boundary,
                split_overlap= split_overlap, remove_punc= remove_punc)

                if len(all_documents['documents']) > 100:
                    warning_msg = ": This might take sometime, please sit back and relax."
                else:
                    warning_msg = ""

                with st.spinner("Running SDG Classification{}".format(warning_msg)):

                    df, x = sdg_classification(haystack_doc=all_documents['documents'],
                                                threshold= threshold)
                    df = df.drop(['Relevancy'], axis = 1)
                    sdg_labels = x.SDG.unique()
                    textrank_keyword_list = []
                    for label in sdg_labels:
                        sdgdata = " ".join(df[df.SDG == label].text.to_list())
                        textranklist_ = textrank(textdata=sdgdata, words= top_n)
                        if len(textranklist_) > 0:
                            textrank_keyword_list.append({'SDG':label, 'TextRank Keywords':",".join(textranklist_)})
                    textrank_keywords_df = pd.DataFrame(textrank_keyword_list)


                    plt.rcParams['font.size'] = 25
                    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
                    # plot
                    fig, ax = plt.subplots()
                    ax.pie(x['count'], colors=colors, radius=2, center=(4, 4),
                        wedgeprops={"linewidth": 1, "edgecolor": "white"},
                        textprops={'fontsize': 14}, 
                        frame=False,labels =list(x.SDG_Num),
                        labeldistance=1.2)
                    # fig.savefig('temp.png', bbox_inches='tight',dpi= 100)
                    

                    st.markdown("#### Anything related to SDGs? ####")

                    c4, c5, c6 = st.columns([1,2,2])

                    with c5:
                        st.pyplot(fig)
                    with c6:
                        labeldf = x['SDG_name'].values.tolist()
                        labeldf = "<br>".join(labeldf)
                        st.markdown(labeldf, unsafe_allow_html=True)
                    st.write("")
                    st.markdown("###### What keywords are present under SDG classified text? ######")

                    AgGrid(textrank_keywords_df, reload_data = False, 
                            update_mode="value_changed",
                    columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)
                    st.write("")
                    st.markdown("###### Top few SDG Classified paragraph/text results ######")

                    AgGrid(df, reload_data = False, update_mode="value_changed",
                    columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)
            else:
                st.info("🤔 No document found, please try to upload it at the sidebar!")
                logging.warning("Terminated as no document provided")


