from haystack.nodes import TransformersQueryClassifier
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes.base import BaseComponent
from haystack.document_stores import InMemoryDocumentStore
import configparser
import streamlit as st
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text
from utils.preprocessing import processingpipeline
from haystack.pipelines import Pipeline

config = configparser.ConfigParser()
config.read_file(open('paramconfig.cfg'))

class QueryCheck(BaseComponent):
    """
    Uses Query Classifier from Haystack, process the query based on query type
    """

    outgoing_edges = 1

    def run(self, query):
        """
        mandatory method to use the cusotm node. Determines the query type, if 
        if the query is of type keyword/statement will modify it to make it more
        useful for sentence transoformers.
        
        """

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


def semanticSearchPipeline(documents:List[Document]):
    """
    creates the semantic search pipeline and document Store object from the
    list of haystack documents. Retriever and Reader model are read from 
    paramconfig. The top_k for the Reader and Retirever are kept same, so that 
    all the results returned by Retriever are used, however the context is 
    extracted by Reader for each retrieved result. The querycheck is added as
    node to process the query.

    
    Params
    ----------
    documents: list of Haystack Documents, returned by preprocessig pipeline.

    Return
    ---------
    semanticsearch_pipeline: Haystack Pipeline object, with all the necessary 
    nodes [QueryCheck, Retriever, Reader]

    document_store: As retriever cna work only with Haystack Document Store, the
    list of document returned by preprocessing pipeline.

    """
    if 'document_store' in st.session_state:
        document_store = st.session_state['document_store']
        temp  = document_store.get_all_documents()
        if st.session_state['filename'] != temp[0].meta['name']:

            document_store = InMemoryDocumentStore()
            document_store.write_documents(documents)
            if 'retriever' in st.session_state:
                retriever = st.session_state['retriever']
                document_store.update_embeddings(retriever)
                # querycheck = 


            # embedding_model = config.get('semantic_search','RETRIEVER')
            # embedding_model_format = config.get('semantic_search','RETRIEVER_FORMAT')
            # embedding_layer = int(config.get('semantic_search','RETRIEVER_EMB_LAYER'))
            # retriever_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
            # retriever = EmbeddingRetriever(
            #     document_store=document_store,
            #     embedding_model=embedding_model,top_k = retriever_top_k,
            #     emb_extraction_layer=embedding_layer, scale_score =True,
            #     model_format=embedding_model_format, use_gpu = True)
            # document_store.update_embeddings(retriever)
        else:
            embedding_model = config.get('semantic_search','RETRIEVER')
            embedding_model_format = config.get('semantic_search','RETRIEVER_FORMAT')
            retriever = EmbeddingRetriever(
                document_store=document_store,
                embedding_model=embedding_model,top_k = retriever_top_k,
                emb_extraction_layer=embedding_layer, scale_score =True,
                model_format=embedding_model_format, use_gpu = True)

    else:
        document_store = InMemoryDocumentStore()
        document_store.write_documents(documents)

        embedding_model = config.get('semantic_search','RETRIEVER')
        embedding_model_format = config.get('semantic_search','RETRIEVER_FORMAT')
        embedding_layer = int(config.get('semantic_search','RETRIEVER_EMB_LAYER'))
        retriever_top_k = int(config.get('semantic_search','RETRIEVER_TOP_K'))
        
        
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=embedding_model,top_k = retriever_top_k,
            emb_extraction_layer=embedding_layer, scale_score =True,
            model_format=embedding_model_format, use_gpu = True)
        st.session_state['retriever'] = retriever
        document_store.update_embeddings(retriever)
        st.session_state['document_store'] = document_store
        querycheck = QueryCheck()
        st.session_state['querycheck'] = querycheck
        reader_model = config.get('semantic_search','READER')
        reader_top_k = retriever_top_k
        reader = FARMReader(model_name_or_path=reader_model,
                        top_k = reader_top_k, use_gpu=True)
        
        st.session_state['reader'] = reader

    querycheck = QueryCheck()
    
    reader_model = config.get('semantic_search','READER')
    reader_top_k = retriever_top_k
    reader = FARMReader(model_name_or_path=reader_model,
                    top_k = reader_top_k, use_gpu=True)
    

    semanticsearch_pipeline = Pipeline()
    semanticsearch_pipeline.add_node(component = querycheck, name = "QueryCheck",
                                    inputs = ["Query"])
    semanticsearch_pipeline.add_node(component = retriever, name = "EmbeddingRetriever",
                                    inputs = ["QueryCheck.output_1"])
    semanticsearch_pipeline.add_node(component = reader, name = "FARMReader",
                                    inputs= ["EmbeddingRetriever"])
    
    return semanticsearch_pipeline, document_store

def semanticsearchAnnotator(matches: List[List[int]], document):
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
        annotated_text = (annotated_text + document[start:start_idx]
                          + str(annotation(body=document[start_idx:end_idx],
                         label="CONTEXT", background="#964448", color='#ffffff')))
        start = end_idx
    
    annotated_text = annotated_text + document[end_idx:]
    
    st.write(
            markdown(annotated_text),
            unsafe_allow_html=True,
        )


def semantic_search(query:Text,documents:List[Document]):
    """
    Performs the Semantic search on the List of haystack documents which is 
    returned by preprocessing Pipeline.

    Params
    -------
    query: Keywords that need to be searche in documents.
    documents: List fo Haystack documents returned by preprocessing pipeline.
    
    """
    semanticsearch_pipeline, doc_store = semanticSearchPipeline(documents)
    results = semanticsearch_pipeline.run(query = query)
    st.markdown("##### Top few semantic search results #####")
    for i,answer in enumerate(results['answers']):
        temp = answer.to_dict()
        start_idx = temp['offsets_in_document'][0]['start']
        end_idx = temp['offsets_in_document'][0]['end']
        match = [[start_idx,end_idx]]
        doc = doc_store.get_document_by_id(temp['document_id']).content
        st.write("Result {}".format(i+1))
        semanticsearchAnnotator(match, doc)