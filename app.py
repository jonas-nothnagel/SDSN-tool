import appStore.keyword_search as keyword_search
import appStore.sdg_analysis as sdg_analysis
import appStore.coherence as coherence
import appStore.info as info
from appStore.multiapp import MultiApp
import streamlit as st

st.set_page_config(page_title = 'Climate Policy Intelligence', 
                   initial_sidebar_state='expanded', layout="wide") 

app = MultiApp()

app.add_app("About","house", info.app)
app.add_app("Search","search", keyword_search.app)
app.add_app("SDG Analysis","gear",sdg_analysis.app)
app.add_app("NDC Comparison","exclude", coherence.app)

app.run()