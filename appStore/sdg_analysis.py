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
from utils.sdg_classifier import runSDGPreprocessingPipeline
from utils.keyword_extraction import keywordExtraction, textrank
import logging
logger = logging.getLogger(__name__)



def app():

    #### APP INFO #####
    with st.container():
        st.markdown("<h2 style='text-align: center; color: black;'> SDG Classification and Keyphrase Extraction </h2>", unsafe_allow_html=True)
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
            keyphrases from the document, calculates a respective relevancy \
            score, and displays the results in a summary table. The keyphrases \
            are extracted using using [Textrank](https://github.com/summanlp/textrank)\
            which is an easy-to-use computational less expensive \
            model leveraging combination of TFIDF and Graph networks.
            """)
        st.markdown("")

    ### Label Dictionary ###
    _lab_dict = {0: 'no_cat',
                1:'SDG 1 - No poverty',
                    2:'SDG 2 - Zero hunger',
                    3:'SDG 3 - Good health and well-being',
                    4:'SDG 4 - Quality education',
                    5:'SDG 5 - Gender equality',
                    6:'SDG 6 - Clean water and sanitation',
                    7:'SDG 7 - Affordable and clean energy',
                    8:'SDG 8 - Decent work and economic growth', 
                    9:'SDG 9 - Industry, Innovation and Infrastructure',
                    10:'SDG 10 - Reduced inequality',
                11:'SDG 11 - Sustainable cities and communities',
                12:'SDG 12 - Responsible consumption and production',
                13:'SDG 13 - Climate action',
                14:'SDG 14 - Life below water',
                15:'SDG 15 - Life on land',
                16:'SDG 16 - Peace, justice and strong institutions',
                17:'SDG 17 - Partnership for the goals',}
    
    ### Main app code ###
    with st.container():
        if st.button("RUN SDG Analysis"):
       
            
            if 'filepath' in st.session_state:
                file_name = st.session_state['filename']
                file_path = st.session_state['filepath']
                allDocuments = runSDGPreprocessingPipeline(file_path,file_name)
                if len(allDocuments['documents']) > 100:
                    warning_msg = ": This might take sometime, please sit back and relax."
                else:
                    warning_msg = ""

                with st.spinner("Running SDG Classification{}".format(warning_msg)):

                    df, x = sdg_classification(allDocuments['documents'])
                    sdg_labels = df.SDG.unique()
                    textrankkeywordlist = []
                    for label in sdg_labels:
                        sdgdata = " ".join(df[df.SDG == label].text.to_list())
                        textranklist_ = textrank(sdgdata)
                        if len(textranklist_) > 0:
                            textrankkeywordlist.append({'SDG':label, 'TextRank Keywords':",".join(textranklist_)})
                    tRkeywordsDf = pd.DataFrame(textrankkeywordlist)


                    plt.rcParams['font.size'] = 25
                    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
                    # plot
                    fig, ax = plt.subplots()
                    ax.pie(x['count'], colors=colors, radius=2, center=(4, 4),
                        wedgeprops={"linewidth": 1, "edgecolor": "white"},
                        textprops={'fontsize': 14}, 
                        frame=False,labels =list(x.SDG),
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

                    AgGrid(tRkeywordsDf, reload_data = False, 
                            update_mode="value_changed",
                    columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)
                    st.write("")
                    st.markdown("###### Top few SDG Classified paragraph/text results ######")

                    AgGrid(df, reload_data = False, update_mode="value_changed",
                    columns_auto_size_mode = ColumnsAutoSizeMode.FIT_CONTENTS)
            else:
                st.info("🤔 No document found, please try to upload it at the sidebar!")
                logging.warning("Terminated as no document provided")


