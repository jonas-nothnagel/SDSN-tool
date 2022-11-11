# set path
import glob, os, sys; 
sys.path.append('../utils')

#import needed libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import docx
from docx.shared import Inches
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
from utils.sdg_classifier import sdg_classification
from utils.sdg_classifier import runSDGPreprocessingPipeline
from utils.keyword_extraction import keywordExtraction, textrank
import logging
logger = logging.getLogger(__name__)



def app():

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
            """
        st.markdown("""**SDG cLassification:** The application assigns paragraphs \
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

        st.markdown("""**Keyphrase Extraction:** The application extracts 15 \
            keyphrases from the document, calculates a respective relevancy \
            score, and displays the results in a summary table. The keyphrases \
            are extracted using using [Textrank](https://github.com/summanlp/textrank)\
            which is an easy-to-use computational less expensive \
            model leveraging combination of TFIDF and Graph networks.
            """)
        st.markdown("")


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
                    # tfidfkeywordList = []
                    textrankkeywordlist = []
                    for label in sdg_labels:
                        sdgdata = " ".join(df[df.SDG == label].text.to_list())
                        # tfidflist_ = keywordExtraction(label,[sdgdata])
                        textranklist_ = textrank(sdgdata, words = 20)
                        tfidfkeywordList.append({'SDG':label, 'TFIDF Keywords':tfidflist_})
                        textrankkeywordlist.append({'SDG':label, 'TextRank Keywords':textranklist_})
                    tfidfkeywordsDf = pd.DataFrame(tfidfkeywordList)
                    tRkeywordsDf = pd.DataFrame(textrankkeywordlist)


                    plt.rcParams['font.size'] = 25
                    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))
                    # plot
                    fig, ax = plt.subplots()
                    ax.pie(x, colors=colors, radius=2, center=(4, 4),
                        wedgeprops={"linewidth": 1, "edgecolor": "white"}, 
                        frame=False,labels =list(x.index))
                    # fig.savefig('temp.png', bbox_inches='tight',dpi= 100)
                    

                    st.markdown("#### Anything related to SDGs? ####")

                    c4, c5, c6 = st.columns([2, 2, 2])

                    with c5:
                        st.pyplot(fig)
                    
                    st.markdown("##### What keywords are present under SDG classified text? #####")
                    st.write("TFIDF BASED")

                    c1, c2, c3 = st.columns([1, 10, 1])
                    with c2:
                        st.table(tfidfkeywordsDf)

                    st.write("TextRank BASED")

                    c11, c12, c13 = st.columns([1, 10, 1])
                    with c12:
                        st.table(tRkeywordsDf)    
                    c7, c8, c9 = st.columns([1, 10, 1])
                    with c8:
                        st.table(df)
            else:
                st.info("🤔 No document found, please try to upload it at the sidebar!")
                logging.warning("Terminated as no document provided")




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