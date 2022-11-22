from haystack.nodes import TfidfRetriever
from haystack.document_stores import InMemoryDocumentStore
import spacy
import re
from spacy.matcher import Matcher
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text, Tuple
from typing_extensions import Literal
from utils.preprocessing import processingpipeline
from utils.streamlitcheck import check_streamlit
import logging
try:
    from termcolor import colored
except:
    pass

try:
    import streamlit as st    
except ImportError:
    logging.info("Streamlit not installed")


def runLexicalPreprocessingPipeline(file_name:str,file_path:str,
                        split_by: Literal["sentence", "word"] = 'word', 
                        split_length:int = 80, split_overlap:int = 0, 
                        remove_punc:bool = False,)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig. As lexical doesnt gets
    affected by overlap, threfore split_overlap = 0 in default paramconfig and 
    split_by = word.

    Params
    ------------

    file_name: filename, in case of streamlit application use 
    st.session_state['filename']
    file_path: filepath, in case of streamlit application use 
    st.session_state['filepath']
    split_by: document splitting strategy either as word or sentence
    split_length: when synthetically creating the paragrpahs from document,
                    it defines the length of paragraph.
    split_overlap: Number of words or sentences that overlap when creating
        the paragraphs. This is done as one sentence or 'some words' make sense
        when  read in together with others. Therefore the overlap is used.
    splititng of text.
    removePunc: to remove all Punctuation including ',' and '.' or not

    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the lexicaal search using TFIDFRetriever we 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """
    
    lexical_processing_pipeline = processingpipeline()


    output_lexical_pre = lexical_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                        "UdfPreProcessor": {"remove_punc": remove_punc, \
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

def runSpacyMatcher(token_list:List[str], document:Text
                    )->Tuple[List[List[int]],spacy.tokens.doc.Doc]:
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

def spacyAnnotator(matches: List[List[int]], document:spacy.tokens.doc.Doc):
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

def lexical_search(query:Text, documents:List[Document],top_k:int):
    """
    Performs the Lexical search on the List of haystack documents which is 
    returned by preprocessing Pipeline.

    Params
    -------
    query: Keywords that need to be searche in documents.
    documents: List of Haystack documents returned by preprocessing pipeline.
    top_k: Number of Top results to be fetched.
    
    """

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    # Haystack Retriever works with document stores only.
    retriever = TfidfRetriever(document_store)
    results = retriever.retrieve(query=query, top_k = top_k)          
    query_tokens = tokenize_lexical_query(query)
    flag = True
    for count, result in enumerate(results):
        matches, doc = runSpacyMatcher(query_tokens,result.content)

        if len(matches) != 0:
            if flag:
                flag = False
                if check_streamlit():
                    st.markdown("##### Top few lexical search (TFIDF) hits #####")
                else:
                    print("Top few lexical search (TFIDF) hits")
            
            if check_streamlit():
                st.write("Result {}".format(count+1))
            else:
                print("Results {}".format(count +1))
            spacyAnnotator(matches, doc)

    if flag:
        if check_streamlit():
            st.info("🤔 No relevant result found. Please try another keyword.")
        else:
            print("No relevant result found. Please try another keyword.")   