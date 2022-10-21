import glob, os, sys; 
sys.path.append('../udfPreprocess')

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
import logging
logger = logging.getLogger(__name__)
import configparser

@st.cache(allow_output_mutation=True)
def load_sdgClassifier():
    classifier = pipeline("text-classification", model= "jonas/sdg_classifier_osdg")
    logging.info("Loading classifier")
    return classifier

def sdg_classification(par_list):
    logging.info("running SDG classifiication")
    config = configparser.ConfigParser()
    config.read_file(open('udfPreprocess/paramconfig.cfg'))
    threshold = float(config.get('sdg','THRESHOLD'))


    classifier = load_sdgClassifier()
    labels = classifier(par_list)
    
    labels_= [(l['label'],l['score']) for l in labels]
    # df2 = DataFrame(labels_, columns=["SDG", "Relevancy"])
    df2 = DataFrame(labels_, columns=["SDG", "Relevancy"])

    df2['text'] = par_list      
    df2 = df2.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
    df2.index += 1
    df2 =df2[df2['Relevancy']>threshold]
    x = df2['SDG'].value_counts()
    df3 = df2.copy()
    df3= df3.drop(['Relevancy'], axis = 1)
    

    return df3, x