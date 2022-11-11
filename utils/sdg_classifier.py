from haystack.nodes import TransformersDocumentClassifier
from haystack.schema import Document
from typing import List, Tuple
import configparser
import logging
from pandas import DataFrame, Series
from utils.preprocessing import processingpipeline
try:
    import streamlit as st
except ImportError:
    logging.info("Streamlit not installed")
config = configparser.ConfigParser()
try:
    config.read_file(open('paramconfig.cfg'))
except Exception:
    logging.info("paramconfig file not found")
    st.info("Please place the paramconfig file in the same directory as app.py")


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



@st.cache(allow_output_mutation=True)
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
    logging.info("Working on SDG Classification")
    threshold = float(config.get('sdg','THRESHOLD'))

    
    classifier = load_sdgClassifier()
    results = classifier.predict(haystackdoc)


    labels_= [(l.meta['classification']['label'],
            l.meta['classification']['score'],l.content,) for l in results]

    df = DataFrame(labels_, columns=["SDG","Relevancy","text"])
    
    df = df.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
    df.index += 1
    df =df[df['Relevancy']>threshold]
    x = df['SDG'].value_counts()
    df= df.drop(['Relevancy'], axis = 1)
    

    return df, x

def runSDGPreprocessingPipeline(filePath, fileName)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig

    Params
    ------------

    file_name: filename, in case of streamlit application use 
    st.session_state['filename']
    file_path: filepath, in case of streamlit application use 
    st.session_state['filepath']


    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the Haysatck implementation of SDG classification we, 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """

    sdg_processing_pipeline = processingpipeline()
    split_by = config.get('sdg','SPLIT_BY')
    split_length = int(config.get('sdg','SPLIT_LENGTH'))
    split_overlap = int(config.get('sdg','SPLIT_OVERLAP'))
    remove_punc = bool(int(config.get('sdg','REMOVE_PUNC')))
    split_respect_sentence_boundary = bool(int(config.get('sdg','RESPECT_SENTENCE_BOUNDARY')))

    output_sdg_pre = sdg_processing_pipeline.run(file_paths = filePath, 
                            params= {"FileConverter": {"file_path": filePath, \
                                        "file_name": fileName}, 
                                     "UdfPreProcessor": {"removePunc": remove_punc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap, \
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})
    
    return output_sdg_pre
