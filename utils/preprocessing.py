from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from haystack.nodes import PDFToTextOCRConverter, PDFToTextConverter
from haystack.nodes import TextConverter, DocxToTextConverter, PreProcessor
from typing import Callable, Dict, List, Optional, Text, Tuple, Union
from typing_extensions import Literal
import pandas as pd
import logging
import re
import string
from haystack.pipelines import Pipeline

def useOCR(file_path: str)-> Text:
    """
    Converts image pdfs into text, Using the Farm-haystack[OCR]
    
    Params
    ----------
    file_path: file_path of uploade file, returned by add_upload function in 
    uploadAndExample.py
    
    Returns the text file as string.
    """

    
    converter = PDFToTextOCRConverter(remove_numeric_tables=True, 
                                      valid_languages=["eng"])
    docs = converter.convert(file_path=file_path, meta=None)
    return docs[0].content




class FileConverter(BaseComponent):
    """
    Wrapper class to convert uploaded document into text by calling appropriate 
    Converter class, will use internally haystack PDFToTextOCR in case of image 
    pdf. Cannot use the FileClassifier from haystack as its doesnt has any 
    label/output class for image.

    1. https://haystack.deepset.ai/pipeline_nodes/custom-nodes
    2. https://docs.haystack.deepset.ai/docs/file_converters
    3. https://github.com/deepset-ai/haystack/tree/main/haystack/nodes/file_converter
    4. https://docs.haystack.deepset.ai/reference/file-converters-api


    """

    outgoing_edges = 1

    def run(self, file_name: str , file_path: str, encoding: Optional[str]=None,
            id_hash_keys: Optional[List[str]] = None,
            ) -> Tuple[dict,str]:
        """ this is required method to invoke the component in 
            the pipeline implementation. 
            
        Params
        ----------
        file_name: name of file
        file_path: file_path of uploade file, returned by add_upload function in 
                    uploadAndExample.py
        
        See the links provided in Class docstring/description to see other params
        
        Return
        ---------
        output: dictionary, with key as identifier and value could be anything 
                we need to return. In this case its the List of Hasyatck Document
        
        output_1: As there is only one outgoing edge, we pass 'output_1' string
        """
        try:
            if file_name.endswith('.pdf'):
                converter = PDFToTextConverter(remove_numeric_tables=True)
            if file_name.endswith('.txt'):
                converter = TextConverter(remove_numeric_tables=True)
            if file_name.endswith('.docx'):
                converter = DocxToTextConverter() 
        except Exception as e:
            logging.error(e)
            return 



        documents = []

        document = converter.convert(
                      file_path=file_path, meta=None, 
                      encoding=encoding, id_hash_keys=id_hash_keys
                      )[0]

        text = document.content

        # if file is image pdf then it will have {'content': "\x0c\x0c\x0c\x0c"}
        # subsitute this substring with '',and check if content is empty string

        text = re.sub(r'\x0c', '', text)
        documents.append(Document(content=text, 
                              meta={"name": file_name}, 
                              id_hash_keys=id_hash_keys))

        
        # check if text is empty and apply pdfOCR converter.
        for i in documents:
            if i.content == "":
                logging.info("Using OCR")
                i.content = useOCR(file_path)
        
        logging.info('file conversion succesful')
        output = {'documents': documents}
        return output, 'output_1'

    def run_batch():
        """
        we dont have requirement to process the multiple files in one go
        therefore nothing here, however to use the custom node we need to have
        this method for the class.
        """
        
        return


def basic(s:str, remove_punc:bool = False):

    """
    Performs basic cleaning of text.

    Params
    ----------
    s: string to be processed
    removePunc: to remove all Punctuation including ',' and '.' or not
    
    Returns: processed string: see comments in the source code for more info
    """
    
    # Remove URLs
    s = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', s, flags=re.MULTILINE)
    s = re.sub(r"http\S+", " ", s)

    # Remove new line characters
    s = re.sub('\n', ' ', s) 

    # Remove punctuations
    if remove_punc == True:
      translator = str.maketrans(' ', ' ', string.punctuation) 
      s = s.translate(translator)
    # Remove distracting single quotes and dotted pattern
    s = re.sub("\'", " ", s)
    s = s.replace("..","") 
    
    return s.strip()


class UdfPreProcessor(BaseComponent):
    """
    class to preprocess the document returned by FileConverter. It will check
    for splitting strategy and splits the document by word or sentences and then
    synthetically create the paragraphs. 

    1. https://docs.haystack.deepset.ai/docs/preprocessor
    2. https://docs.haystack.deepset.ai/reference/preprocessor-api
    3. https://github.com/deepset-ai/haystack/tree/main/haystack/nodes/preprocessor

    """
    outgoing_edges = 1

    def run(self, documents:List[Document], remove_punc:bool=False, 
            split_by: Literal["sentence", "word"] = 'sentence',
            split_length:int = 2, split_respect_sentence_boundary:bool = False,
            split_overlap:int = 0):

        """ this is required method to invoke the component in 
        the pipeline implementation. 
            
        Params
        ----------
        documents: documents from the output dictionary returned by Fileconverter
        remove_punc: to remove all Punctuation including ',' and '.' or not
        split_by: document splitting strategy either as word or sentence
        split_length: when synthetically creating the paragrpahs from document,
                      it defines the length of paragraph.
        split_respect_sentence_boundary: Used when using 'word' strategy for 
        splititng of text.
        split_overlap: Number of words or sentences that overlap when creating
        the paragraphs. This is done as one sentence or 'some words' make sense
        when  read in together with others. Therefore the overlap is used.
        
        Return
        ---------
        output: dictionary, with key as identifier and value could be anything 
                we need to return. In this case the output will contain 4 objects
                the paragraphs text list as List, Haystack document, Dataframe and 
                one raw text file.
        
        output_1: As there is only one outgoing edge, we pass 'output_1' string
      
        """
        
        if split_by == 'sentence':
            split_respect_sentence_boundary = False
    
        else:
            split_respect_sentence_boundary = split_respect_sentence_boundary
      
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by=split_by,
            split_length=split_length,
            split_respect_sentence_boundary= split_respect_sentence_boundary,
            split_overlap=split_overlap,

            # will add page number only in case of PDF not for text/docx file.
            add_page_number=True
            )
        
        for i in documents:
            # # basic cleaning before passing it to preprocessor.
            # i = basic(i)
            docs_processed = preprocessor.process([i])
            for item in docs_processed:
                item.content = basic(item.content, remove_punc= remove_punc)

        df = pd.DataFrame(docs_processed)
        all_text = " ".join(df.content.to_list())
        para_list = df.content.to_list()
        logging.info('document split into {} paragraphs'.format(len(para_list)))
        output = {'documents': docs_processed,
                  'dataframe': df,
                  'text': all_text,
                  'paraList': para_list
                 }
        return output, "output_1"
    def run_batch():
        """
            we dont have requirement to process the multiple files in one go
            therefore nothing here, however to use the custom node we need to have
            this method for the class.
        """
        return

def processingpipeline():
    """
    Returns the preprocessing pipeline. Will use FileConverter and UdfPreProcesor 
    from utils.preprocessing

    """

    preprocessing_pipeline = Pipeline()
    file_converter = FileConverter()
    custom_preprocessor = UdfPreProcessor()

    preprocessing_pipeline.add_node(component=file_converter, 
                                    name="FileConverter", inputs=["File"])
    preprocessing_pipeline.add_node(component = custom_preprocessor, 
                            name ='UdfPreProcessor', inputs=["FileConverter"])

    return preprocessing_pipeline

