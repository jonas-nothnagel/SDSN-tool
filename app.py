import appStore.keyword_search as keyword_search
import appStore.sdg_analysis as sdg_analysis
import appStore.coherence as coherence
import appStore.info as info
from appStore.multiapp import MultiApp
import streamlit as st

# This branch is before the download option was implemented
st.set_page_config(f'SDSN x GIZ Policy Action Tracking v0.1', layout="wide")

app = MultiApp()

app.add_app("SDG Analysis", sdg_analysis.app)
app.add_app("Search", keyword_search.app)
app.add_app("NDC Coherence", coherence.app)
app.add_app("About", info.app)

app.run()