from tkinter import Text
from haystack.nodes import TransformersDocumentClassifier
from haystack.schema import Document
from typing import List, Tuple
import configparser
import streamlit as st
from pandas import DataFrame, Series
import logging
from udfPreprocess.preprocessing import processingpipeline
config = configparser.ConfigParser()
config.read_file(open('udfPreprocess/paramconfig.cfg'))

@st.cache(allow_output_mutation=True)
def load_sdgClassifier():
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.
     1. https://docs.haystack.deepset.ai/reference/document-classifier-api
     2. https://docs.haystack.deepset.ai/docs/document_classifier

    Return: document classifier model
    """
    logging.info("Loading classifier")
    doc_classifier_model = config.get('sdg','MODEL')
    doc_classifier = TransformersDocumentClassifier(
        model_name_or_path=doc_classifier_model,
        task="text-classification")
    return doc_classifier


def sdg_classification(haystackdoc:List[Document])->Tuple[DataFrame,Series]:
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
    logging.info("running SDG classifiication")
    threshold = float(config.get('sdg','THRESHOLD'))


    classifier = load_sdgClassifier()
    results = classifier.predict(haystackdoc)

    
    labels_= [(l.meta['classification']['label'],
               l.meta['classification']['score'],l.content,) for l in results]

    df = DataFrame(labels_, columns=["text","SDG","Relevancy"])

    # df['text'] = paraList      
    df = df.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
    df.index += 1
    df =df[df['Relevancy']>threshold]
    x = df['SDG'].value_counts()
    #  df = df.copy()
    df= df.drop(['Relevancy'], axis = 1)
    

    return df, x

def runSDGPreprocessingPipeline()->List[Text]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig
    
    """
    file_path = st.session_state['filepath']
    file_name = st.session_state['filename']
    sdg_processing_pipeline = processingpipeline()
    split_by = config.get('sdg','SPLIT_BY')
    split_length = int(config.get('sdg','SPLIT_LENGTH'))

    output_sdg_pre = sdg_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                     "UdfPreProcessor": {"removePunc": False, \
                                            "split_by": split_by, \
                                            "split_length":split_length}})
    
    return output_sdg_pre['documents']
