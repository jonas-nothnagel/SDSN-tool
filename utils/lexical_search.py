from haystack.nodes import TfidfRetriever
from haystack.document_stores import InMemoryDocumentStore
import spacy
import re
from spacy.matcher import Matcher
import streamlit as st
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text
from utils.preprocessing import processingpipeline
from utils.streamlitcheck import check_streamlit
import configparser
import logging
try:
    from termcolor import colored
except:
    pass

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


def runLexicalPreprocessingPipeline(file_path, file_name)->List[Document]:
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
    has four objects. For the lexicaal search using TFIDFRetriever we 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """
    
    lexical_processing_pipeline = processingpipeline()
    split_by = config.get('lexical_search','SPLIT_BY')
    split_length = int(config.get('lexical_search','SPLIT_LENGTH'))
    split_overlap = int(config.get('lexical_search','SPLIT_OVERLAP'))

    output_lexical_pre = lexical_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                        "UdfPreProcessor": {"removePunc": False, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap}})

    return output_lexical_pre


def tokenize_lexical_query(query:str)-> List[str]:
    """
    Removes the stop words from query and returns the list of important keywords
    in query. For the lexical search the relevent paragraphs in document are 
    retreived using TfIDFretreiver from Haystack. However to highlight these 
    keywords we need the tokenized form of query.

    Params
    --------
    query: string which represents either list of keywords user is looking for 
            or a query in form of Question.
    
    Return
    -----------
    token_list: list of important keywords in the query.
     
    """
    nlp = spacy.load("en_core_web_sm")    
    token_list = [token.text.lower() for token in nlp(query) 
                  if not (token.is_stop or token.is_punct)]
    return token_list

def runSpacyMatcher(token_list:List[str], document:Text):
    """
    Using the spacy in backend finds the keywords in the document using the 
    Matcher class from spacy. We can alternatively use the regex, but spacy
    finds all keywords in serialized manner which helps in annotation of answers.

    Params
    -------
    token_list: this is token list which tokenize_lexical_query function returns
    document: text in which we need to find the tokens

    Return
    --------
    matches: List of [start_index, end_index] in the spacydoc(at word level not 
    character) for the keywords in token list.

    spacydoc: the keyword index in the spacydoc are at word level and not character,
    therefore to allow the annotator to work seamlessly we return the spacydoc.

    """
    nlp = spacy.load("en_core_web_sm")
    spacydoc = nlp(document)
    matcher = Matcher(nlp.vocab)
    token_pattern = [[{"LOWER":token}] for token in token_list]
    matcher.add(",".join(token_list), token_pattern)
    spacymatches = matcher(spacydoc)

    # getting start and end index in spacydoc so that annotator can work seamlessly
    matches = []
    for match_id, start, end in spacymatches:
        matches = matches + [[start, end]]
    
    return matches, spacydoc

def runRegexMatcher(token_list:List[str], document:Text):
    """
    Using the regex in backend finds the keywords in the document.

    Params
    -------
    token_list: this is token list which tokenize_lexical_query function returns

    document: text in which we need to find the tokens

    Return
    --------
    matches: List of [start_index, end_index] in the document for the keywords 
    in token list at character level.

    document: the keyword index returned by regex are at character level,
    therefore to allow the annotator to work seamlessly we return the text back.

    """
    matches = []
    for token in token_list:
        matches = (matches + 
                  [[val.start(), val.start() + 
                  len(token)] for val in re.finditer(token, document)])
    
    return matches, document

def spacyAnnotator(matches: List[List[int]], document:spacy.token.doc.Doc):
    """
    This is spacy Annotator and needs spacy.doc
    Annotates the text in the document defined by list of [start index, end index]
    Example: "How are you today", if document type is text, matches = [[0,3]]
    will give answer = "How", however in case we used the spacy matcher then the
    matches = [[0,3]] will give answer = "How are you". However if spacy is used
    to find "How" then the matches = [[0,1]] for the string defined above.

    Params
    -----------
    matches: As mentioned its list of list. Example [[0,1],[10,13]]
    document: document which needs to be indexed.


    Return
    --------
    will send the output to either app front end using streamlit or 
    write directly to output screen.

    """
    start = 0
    annotated_text = ""
    for match in matches:
        start_idx = match[0]
        end_idx = match[1]

        if check_streamlit():
            annotated_text = (annotated_text + document[start:start_idx].text 
                            + str(annotation(body=document[start_idx:end_idx].text,
                            label="ANSWER", background="#964448", color='#ffffff')))
        else:
            annotated_text = (annotated_text + document[start:start_idx].text
                            + colored(document[start_idx:end_idx].text,
                          "green", attrs = ['bold']))


        start = end_idx
    
    annotated_text = annotated_text + document[end_idx:].text


    if check_streamlit():

        st.write(
                markdown(annotated_text),
                unsafe_allow_html=True,
            )
    else:
        print(annotated_text)

def lexical_search(query:Text,documents:List[Document]):
    """
    Performs the Lexical search on the List of haystack documents which is 
    returned by preprocessing Pipeline.

    Params
    -------
    query: Keywords that need to be searche in documents.
    documents: List of Haystack documents returned by preprocessing pipeline.
    
    """

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    # Haystack Retriever works with document stores only.
    retriever = TfidfRetriever(document_store)
    results = retriever.retrieve(query=query, 
                            top_k= int(config.get('lexical_search','TOP_K')))
    query_tokens = tokenize_lexical_query(query)
    for count, result in enumerate(results):
        matches, doc = runSpacyMatcher(query_tokens,result.content)
        if len(matches) != 0:
            if check_streamlit():
                st.write("Result {}".format(count+1))
            else:
                print("Results {}".format(count +1))
            spacyAnnotator(matches, doc)
        