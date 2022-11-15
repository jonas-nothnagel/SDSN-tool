from haystack.nodes import TransformersQueryClassifier
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes.base import BaseComponent
from haystack.document_stores import InMemoryDocumentStore
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text
from typing_extensions import Literal
from utils.preprocessing import processingpipeline
from utils.streamlitcheck import check_streamlit
from haystack.pipelines import Pipeline
import logging
try:
    from termcolor import colored
except:
    pass
try:
    import streamlit as st    
except ImportError:
    logging.info("Streamlit not installed")


@st.cache(allow_output_mutation=True)
def loadQueryClassifier():
    """
    retuns the haystack query classifier model
    model = shahrukhx01/bert-mini-finetune-question-detection
    
    """
    query_classifier = TransformersQueryClassifier(model_name_or_path=
                            "shahrukhx01/bert-mini-finetune-question-detection")
    return query_classifier

class QueryCheck(BaseComponent):
    """
    Uses Query Classifier from Haystack, process the query based on query type.
    Ability to determine the statements is not so good, therefore the chances 
    statement also get modified. Ex: "List water related issues" will be 
    identified by the model as keywords, and therefore it be processed as "find 
    all issues related to 'list all water related issues'". This is one shortcoming
    but is igonred for now, as semantic search will not get affected a lot, by this.

    1. https://docs.haystack.deepset.ai/docs/query_classifier

    """

    outgoing_edges = 1

    def run(self, query):
        """
        mandatory method to use the cusotm node. Determines the query type, if 
        if the query is of type keyword/statement will modify it to make it more
        useful for sentence transoformers.
        
        """
        query_classifier = loadQueryClassifier()
        result = query_classifier.run(query=query)

        if result[1] == "output_1":
            output = {"query":query,
                       "query_type": 'question/statement'}
        else:
            output = {"query": "find all issues related to {}".format(query),
                      "query_type": 'statements/keyword'}
        logging.info(output)
        return output, "output_1"
    
    def run_batch(self, query):
        pass

@st.cache(allow_output_mutation=True)
def runSemanticPreprocessingPipeline(file_path, file_name, 
                split_by: Literal["sentence", "word"] = 'sentence',
                split_respect_sentence_boundary = False,
                split_length:int = 2, split_overlap = 0,
                removePunc = False)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline.

    Params
    ------------

    file_name: filename, in case of streamlit application use 
    st.session_state['filename']
    file_path: filepath, in case of streamlit application use 
    st.session_state['filepath']
    removePunc: to remove all Punctuation including ',' and '.' or not
    split_by: document splitting strategy either as word or sentence
    split_length: when synthetically creating the paragrpahs from document,
                    it defines the length of paragraph.
    split_respect_sentence_boundary: Used when using 'word' strategy for 
    splititng of text.

    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the Haysatck implementation of semantic search we, 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """

    semantic_processing_pipeline = processingpipeline()

    output_semantic_pre = semantic_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                        "UdfPreProcessor": {"removePunc": removePunc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap,
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})

    return output_semantic_pre


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},allow_output_mutation=True)
def loadRetriever(embedding_model:Text =  None, embedding_model_format:Text = None, 
                 embedding_layer:int = None,  retriever_top_k:int = 10, 
                 document_store:InMemoryDocumentStore = None):
    """
    Returns the Retriever model based on params provided.
    1. https://docs.haystack.deepset.ai/docs/retriever#embedding-retrieval-recommended
    2. https://www.sbert.net/examples/applications/semantic-search/README.html
    3. https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py

    
    Params
    ---------
    embedding_model: Name of the model to be used for embedding. Check the links
    provided in documentation
    embedding_model_format: check the github link of Haystack provided in documentation
    embedding_layer: check the github link of Haystack provided in documentation
    retriever_top_k: Number of Top results to be returned by retriever
    document_store: InMemoryDocumentStore, write haystack Document list to DocumentStore
    and pass the same to function call. Can be done using createDocumentStore from utils.
    
    Return
    -------
    retriever: embedding model
    """
    logging.info("loading retriever")
    if document_store is None:
        logging.warning("Retriever initialization requires the DocumentStore")
        return
    
    retriever = EmbeddingRetriever(
                embedding_model=embedding_model,top_k = retriever_top_k,
                document_store = document_store,
                emb_extraction_layer=embedding_layer, scale_score =True,
                model_format=embedding_model_format, use_gpu = True)
    if check_streamlit:
        st.session_state['retriever'] = retriever
    return retriever

@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},allow_output_mutation=True)
def createDocumentStore(documents:List[Document], similarity:str = 'dot_product', 
                            embedding_dim:int = 768):
    """
    Creates the InMemory Document Store from haystack list of Documents.
    It is  mandatory component for Retriever to work in Haystack frame work.
    
    Params
    -------
    documents: List of haystack document. If using the preprocessing pipeline, 
    can be fetched key = 'documents; on output of preprocessing pipeline.
    similarity: scoring function, can be either 'cosine' or 'dot_product'
    embedding_dim: Document store has default value of embedding size = 768, and
    update_embeddings method of Docstore cannot infer the embedding size of 
    retiever automaticallu, therefore set this value as per the model card.
    
    Return
    -------
    document_store: InMemory Document Store object type.
    
    """
    document_store = InMemoryDocumentStore(similarity = similarity, 
                                        embedding_dim = embedding_dim )
    document_store.write_documents(documents)
    
    return document_store


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},allow_output_mutation=True)
def semanticSearchPipeline(documents:List[Document], embedding_model:Text =  None, 
                useQueryCheck = True, embedding_model_format:Text = None, 
                 embedding_layer:int = None,  retriever_top_k:int = 10,
                 reader_model:str =  None, reader_top_k:int = 10,
                 embedding_dim:int = 768):
    """
    creates the semantic search pipeline and document Store object from the
    list of haystack documents. The top_k for the Reader and Retirever are kept  
    same, so that all the results returned by Retriever are used, however the 
    context is extracted by Reader for each retrieved result. The querycheck is 
    added as node to process the query.
    1. https://docs.haystack.deepset.ai/docs/retriever#embedding-retrieval-recommended
    2. https://www.sbert.net/examples/applications/semantic-search/README.html
    3. https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py
    4. https://docs.haystack.deepset.ai/docs/reader


    
    Params
    ----------
    documents: list of Haystack Documents, returned by preprocessig pipeline.
    embedding_model: Name of the model to be used for embedding. Check the links
    provided in documentation
    embedding_model_format: check the github link of Haystack provided in documentation
    embedding_layer: check the github link of Haystack provided in documentation
    retriever_top_k: Number of Top results to be returned by retriever
    reader_model: Name of the model to be used for Reader node in hasyatck 
    Pipeline. Check the links provided in documentation
    reader_top_k: Reader will use retrieved results to further find better matches.
                As purpose here is to use reader to extract context, the value is
                same as retriever_top_k.
    useQueryCheck: Whether to use the querycheck which modifies the query or not.
    embedding_dim: Document store has default value of embedding size = 768, and
    update_embeddings method of Docstore cannot infer the embedding size of 
    retiever automaticallu, therefore set this value as per the model card.


    Return
    ---------
    semanticsearch_pipeline: Haystack Pipeline object, with all the necessary 
    nodes [QueryCheck, Retriever, Reader]

    document_store: As retriever can work only with Haystack Document Store, the
    list of document returned by preprocessing pipeline are fed into to get 
    InMemmoryDocumentStore object type, with retriever updating the embedding
    embeddings of each paragraph in document store.

    """
    document_store = createDocumentStore(documents=documents, 
                                    embedding_dim=embedding_dim)                  
    retriever = loadRetriever(embedding_model = embedding_model,
                    embedding_model_format=embedding_model_format,
                    embedding_layer=embedding_layer,  
                    retriever_top_k= retriever_top_k, 
                    document_store = document_store)
                
    document_store.update_embeddings(retriever)
    reader = FARMReader(model_name_or_path=reader_model,
                    top_k = reader_top_k, use_gpu=True)
    semanticsearch_pipeline = Pipeline()
    if useQueryCheck:
        querycheck = QueryCheck()
        semanticsearch_pipeline.add_node(component = querycheck, name = "QueryCheck",
                                        inputs = ["Query"])
        semanticsearch_pipeline.add_node(component = retriever, name = "EmbeddingRetriever",
                                        inputs = ["QueryCheck.output_1"])
        semanticsearch_pipeline.add_node(component = reader, name = "FARMReader",
                                        inputs= ["EmbeddingRetriever"])
    else:
        semanticsearch_pipeline.add_node(component = retriever, name = "EmbeddingRetriever",
                                        inputs = ["Query"])
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
        if check_streamlit():
            annotated_text = (annotated_text + document[start:start_idx]
                            + str(annotation(body=document[start_idx:end_idx],
                            label="Context", background="#964448", color='#ffffff')))
        else:
            annotated_text = (annotated_text + document[start:start_idx]
                            + colored(document[start_idx:end_idx],
                          "green", attrs = ['bold']))
        start = end_idx
    
    annotated_text = annotated_text + document[end_idx:]

    if check_streamlit():

        st.write(
                markdown(annotated_text),
                unsafe_allow_html=True,
            )
    else:
        print(annotated_text)
    

def semantic_search(query:Text,documents:List[Document],embedding_model:Text, 
                embedding_model_format:Text, 
                 embedding_layer:int,  reader_model:str,
                 retriever_top_k:int = 10, reader_top_k:int = 10,
                 return_results:bool = False, embedding_dim:int = 768):
    """
    Performs the Semantic search on the List of haystack documents which is 
    returned by preprocessing Pipeline.

    Params
    -------
    query: Keywords that need to be searche in documents.
    documents: List fo Haystack documents returned by preprocessing pipeline.
    
    """
    semanticsearch_pipeline, doc_store = semanticSearchPipeline(documents,
                        embedding_model= embedding_model, 
                        embedding_layer= embedding_layer,
                        embedding_model_format= embedding_model_format,
                        reader_model= reader_model, retriever_top_k= retriever_top_k,
                        reader_top_k= reader_top_k, embedding_dim=embedding_dim)

    results = semanticsearch_pipeline.run(query = query)
    if return_results:
        return results
    else:
        if check_streamlit:
            st.markdown("##### Top few semantic search results #####")
        else:
            print("Top few semantic search results")
        for i,answer in enumerate(results['answers']):
            temp = answer.to_dict()
            start_idx = temp['offsets_in_document'][0]['start']
            end_idx = temp['offsets_in_document'][0]['end']
            match = [[start_idx,end_idx]]
            doc = doc_store.get_document_by_id(temp['document_id']).content
            if check_streamlit:
                st.write("Result {}".format(i+1))
            else:
                print("Result {}".format(i+1))
            semanticsearchAnnotator(match, doc)