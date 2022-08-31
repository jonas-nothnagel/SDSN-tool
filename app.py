import appStore.keyword_search as keyword_search
import appStore.sdg_analysis as sdg_analysis
import appStore.coherence as coherence
import appStore.info as info
from appStore.multiapp import MultiApp
import streamlit as st

st.set_page_config(f'SDSN x GIZ Policy Action Tracking v0.1', layout="wide")

app = MultiApp()

app.add_app("Analyse Policy Document", sdg_analysis.app)
app.add_app("KeyWord Search", keyword_search.app)
app.add_app("Check Coherence", coherence.app)
app.add_app("Info", info.app)

app.run()