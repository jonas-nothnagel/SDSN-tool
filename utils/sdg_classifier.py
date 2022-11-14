from haystack.nodes import TransformersDocumentClassifier
from haystack.schema import Document
from typing import List, Tuple, Float
from typing_extensions import Literal
import configparser
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.checkconfig import getconfig
from utils.preprocessing import processingpipeline
try:
    import streamlit as st
except ImportError:
    logging.info("Streamlit not installed")

@st.cache(allow_output_mutation=True)
def load_sdgClassifier(configFile = None, docClassifierModel = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    model should be passed.
    1. https://docs.haystack.deepset.ai/reference/document-classifier-api
    2. https://docs.haystack.deepset.ai/docs/document_classifier

    Params
    --------
    configFile: config file from which to read the model name
    docClassifierModel: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.


    Return: document classifier model
    """
    if not docClassifierModel:
        if not configFile:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(configFile)
            docClassifierModel = config.get('sdg','MODEL')
    
    logging.info("Loading classifier")    
    doc_classifier = TransformersDocumentClassifier(
                        model_name_or_path=docClassifierModel,
                        task="text-classification")

    return doc_classifier
        

@st.cache(allow_output_mutation=True)
def sdg_classification(haystackdoc:List[Document],
                        threshold:float, classifiermodel)->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate label for each text. these labels are in terms of if text 
    belongs to which particular Sustainable Devleopment Goal (SDG).

    Params
    ---------
    haystackdoc: List of haystack Documents. The output of Preprocessing Pipeline 
    contains the list of paragraphs in different format,here the list of 
    Haystack Documents is used.

    Returns
    ----------
    df: Dataframe with two columns['SDG:int', 'text']
    x: Series object with the unique SDG covered in the document uploaded and 
    the number of times it is covered/discussed/count_of_paragraphs. 

    """
    logging.info("Working on SDG Classification")
    results = classifiermodel.predict(haystackdoc)


    labels_= [(l.meta['classification']['label'],
            l.meta['classification']['score'],l.content,) for l in results]

    df = DataFrame(labels_, columns=["SDG","Relevancy","text"])
    
    df = df.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
    df.index += 1
    df =df[df['Relevancy']>threshold]

    # creating the dataframe for value counts of SDG, along with 'title' of SDGs
    x = df['SDG'].value_counts()
    x = x.rename('count')
    x = x.rename_axis('SDG').reset_index()
    x["SDG"] = pd.to_numeric(x["SDG"])
    x = x.sort_values(by=['count'])
    x['SDG_name'] = x['SDG'].apply(lambda x: _lab_dict[x])
    x['SDG_Num'] = x['SDG'].apply(lambda x: "SDG "+str(x))

    df['SDG'] = pd.to_numeric(df['SDG'])
    df = df.sort_values('SDG')
    
    return df, x

def runSDGPreprocessingPipeline(filePath, fileName, 
            split_by: Literal["sentence", "word"] = 'sentence',
            split_respect_sentence_boundary = False,
            split_length:int = 2, split_overlap = 0,
            removePunc = False)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig

    Params
    ------------

    file_name: filename, in case of streamlit application use 
    st.session_state['filename']
    file_path: filepath, in case of streamlit application use 
    removePunc: to remove all Punctuation including ',' and '.' or not
    split_by: document splitting strategy either as word or sentence
    split_length: when synthetically creating the paragrpahs from document,
                    it defines the length of paragraph.
    split_respect_sentence_boundary: Used when using 'word' strategy for 
    splititng of text.


    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the Haysatck implementation of SDG classification we, 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """

    sdg_processing_pipeline = processingpipeline()

    output_sdg_pre = sdg_processing_pipeline.run(file_paths = filePath, 
                            params= {"FileConverter": {"file_path": filePath, \
                                        "file_name": fileName}, 
                                     "UdfPreProcessor": {"removePunc": removePunc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap, \
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})
    
    return output_sdg_pre
