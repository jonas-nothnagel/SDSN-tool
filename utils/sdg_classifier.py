from haystack.nodes import TransformersDocumentClassifier
from haystack.schema import Document
from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.checkconfig import getconfig
from utils.streamlitcheck import check_streamlit
from utils.preprocessing import processingpipeline
try:
    import streamlit as st
except ImportError:
    logging.info("Streamlit not installed")

## Labels dictionary ###
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

@st.cache(allow_output_mutation=True)
def load_sdgClassifier(config_file = None, classifier_name = None):
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
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('sdg','MODEL')
    
    logging.info("Loading classifier")    
    doc_classifier = TransformersDocumentClassifier(
                        model_name_or_path=classifier_name,
                        task="text-classification")

    return doc_classifier
        

@st.cache(allow_output_mutation=True)
def sdg_classification(haystack_doc:List[Document],
                        threshold:float, classifier_model= None)->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate label for each text. these labels are in terms of if text 
    belongs to which particular Sustainable Devleopment Goal (SDG).

    Params
    ---------
    haystackdoc: List of haystack Documents. The output of Preprocessing Pipeline 
    contains the list of paragraphs in different format,here the list of 
    Haystack Documents is used.
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly, however in case of
    streamlit avoid it.


    Returns
    ----------
    df: Dataframe with two columns['SDG:int', 'text']
    x: Series object with the unique SDG covered in the document uploaded and 
    the number of times it is covered/discussed/count_of_paragraphs. 

    """
    logging.info("Working on SDG Classification")
    if not classifier_model:
        if check_streamlit:
            classifier_model = st.session_state['sdg_classifier']
        else:
            logging.warning("No streamlit envinornment found, Pass the classifier")
            return
    
    results = classifier_model.predict(haystack_doc)


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
            remove_punc = False)->List[Document]:
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
                                     "UdfPreProcessor": {"remove_punc": remove_punc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap, \
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})
    
    return output_sdg_pre
