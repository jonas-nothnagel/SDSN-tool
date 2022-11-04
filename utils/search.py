from haystack.nodes import TfidfRetriever, TransformersQueryClassifier
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes.base import BaseComponent
from haystack.document_stores import InMemoryDocumentStore
import configparser
import spacy
import re
from spacy.matcher import Matcher
import streamlit as st
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text
from utils.preprocessing import processingpipeline
from haystack.pipelines import Pipeline

config = configparser.ConfigParser()
config.read_file(open('paramconfig.cfg'))


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

def searchAnnotator(matches: List[List[int]], document):
    """
    Annotates the text in the document defined by list of [start index, end index]
    Example: "How are you today", if document type is text, matches = [[0,3]]
    will give answer = "How", however in case we used the spacy matcher then the
    matches = [[0,3]] will give answer = "How are you". However if spacy is used
    to find "How" then the matches = [[0,1]] for the string defined above.

    """
    start = 0
    annotated_text = ""
    for match in matches:
        start_idx = match[0]
        end_idx = match[1]
        annotated_text = (annotated_text + document[start:start_idx].text 
                          + str(annotation(body=document[start_idx:end_idx].text,
                         label="ANSWER", background="#964448", color='#ffffff')))
        start = end_idx
    
    annotated_text = annotated_text + document[end_idx:].text
    
    st.write(
            markdown(annotated_text),
            unsafe_allow_html=True,
        )

def lexical_search(query:Text,documents:List[Document]):
    """
    Performs the Lexical search on the List of haystack documents which is 
    returned by preprocessing Pipeline.
    """

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    
    # Haystack Retriever works with document stores only.
    retriever = TfidfRetriever(document_store)
    results = retriever.retrieve(query=query, 
                            top_k= int(config.get('lexical_search','TOP_K')))
    query_tokens = tokenize_lexical_query(query)
    for count, result in enumerate(results):
        # if result.content != "":
        matches, doc = runSpacyMatcher(query_tokens,result.content)
        if len(matches) != 0:
            st.write("Result {}".format(count+1))
            searchAnnotator(matches, doc)

def runLexicalPreprocessingPipeline()->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig

    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the lexicaal search using TFIDFRetriever we 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """
    file_path = st.session_state['filepath']
    file_name = st.session_state['filename']
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

    return output_lexical_pre['documents']   
        
def runSemanticPreprocessingPipeline()->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig

    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the Haysatck implementation of semantic search we, 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """
    file_path = st.session_state['filepath']
    file_name = st.session_state['filename']
    semantic_processing_pipeline = processingpipeline()
    split_by = config.get('semantic_search','SPLIT_BY')
    split_length = int(config.get('semantic_search','SPLIT_LENGTH'))
    split_overlap = int(config.get('semantic_search','SPLIT_OVERLAP'))

    output_semantic_pre = semantic_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                        "UdfPreProcessor": {"removePunc": False, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap}})

    return output_semantic_pre['documents']

class QueryCheck(BaseComponent):

    outgoing_edges = 1

    def run(self, query):

        query_classifier =  TransformersQueryClassifier(model_name_or_path=
                            "shahrukhx01/bert-mini-finetune-question-detection")

        
        result = query_classifier.run(query=query)

        if result[1] == "output_1":
            output = {"query":query,
                       "query_type": 'question/statement'}
        else:
            output = {"query": "find all issues related to {}".format(query),
                      "query_type": 'statements/keyword'}

        return output, "output_1"
    
    def run_batch(self, query):
        pass


def semanticSearchPipeline(documents, show_answers = False):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    embedding_model = config.get('semantic_search','RETRIEVER')
    embedding_model_format = config.get('semantic_search','RETRIEVER_FORMAT')
    embedding_layer = int(config.get('semantic_search','RETRIEVER_EMB_LAYER'))
    retriever_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
    

    
    querycheck = QueryCheck()
    retriever = EmbeddingRetriever(
                document_store=document_store,
                embedding_model=embedding_model,top_k = retriever_top_k,
                emb_extraction_layer=embedding_layer, scale_score =True,
                model_format=embedding_model_format, use_gpu = True)
    document_store.update_embeddings(retriever)
    

    semanticsearch_pipeline = Pipeline()
    semanticsearch_pipeline.add_node(component = querycheck, name = "QueryCheck",
                                    inputs = ["Query"])
    semanticsearch_pipeline.add_node(component = retriever, name = "EmbeddingRetriever",
                                    inputs = ["QueryCheck.output_1"])
    if show_answers == True:
        reader_model = config.get('semantic_search','READER')
        reader_top_k = retriever_top_k
        reader = FARMReader(model_name_or_path=reader_model,
                        top_k = reader_top_k, use_gpu=True)
    
        semanticsearch_pipeline.add_node(component = reader, name = "FARMReader",
                                        inputs= ["EmbeddingRetriever"])
    
    return semanticsearch_pipeline, document_store

def semantic_search(query:Text,documents:List[Document],show_answers = False):
    """
    Performs the Lexical search on the List of haystack documents which is 
    returned by preprocessing Pipeline.
    """
    threshold = 0.4
    semanticsearch_pipeline, doc_store = semanticSearchPipeline(documents, 
                                                    show_answers=show_answers)
    results = semanticsearch_pipeline.run(query = query)
    
    
    if show_answers == False:
        results = results['documents']
        for i,queryhit in enumerate(results):
                            
            if queryhit.score > threshold:
                st.write("\t {}: \t {}".format(i+1, queryhit.content.replace("\n", " ")))
                st.markdown("---")
        
    else:
        matches = []
        doc = []
        for answer in results['answers']:
            if answer.score >0.01:
                temp = answer.to_dict()
                start_idx = temp['offsets_in_document'][0]['start']
                end_idx = temp['offsets_in_document'][0]['end']

                matches.append([start_idx,end_idx])
                doc.append(doc_store.get_document_by_id(temp['document_id']).content)
        searchAnnotator(matches,doc)
        


            

            


    return results


