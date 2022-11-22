from haystack.nodes import TransformersQueryClassifier, Docs2Answers
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.nodes.base import BaseComponent
from haystack.document_stores import InMemoryDocumentStore
from markdown import markdown
from annotated_text import annotation
from haystack.schema import Document
from typing import List, Text, Union
from typing_extensions import Literal
from utils.preprocessing import processingpipeline
from utils.streamlitcheck import check_streamlit
from haystack.pipelines import Pipeline
import pandas as pd
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
    identified by the model as keywords, and therefore it be processed as "what 
    are the 'list all water related issues' related issues and discussions?". 
    This is one shortcoming but is igonred for now, as semantic search will not 
    get affected a lot, by this. If you want to pass keywords list and want to 
    do batch processing use. run_batch. Example: if you want to find relevant 
    passages for water, food security, poverty then querylist = ["water", "food 
    security","poverty"] and then execute QueryCheck.run_batch(queries = querylist)

    1. https://docs.haystack.deepset.ai/docs/query_classifier

    """

    outgoing_edges = 1

    def run(self, query:str):
        """
        mandatory method to use the custom node. Determines the query type, if 
        if the query is of type keyword/statement will modify it to make it more
        useful for sentence transoformers.

        Params
        --------
        query: query/statement/keywords in form of string

        Return
        ------
        output: dictionary, with key as identifier and value could be anything 
                we need to return. In this case the output contain key = 'query'.
        
        output_1: As there is only one outgoing edge, we pass 'output_1' string
        
        """
        query_classifier = loadQueryClassifier()
        result = query_classifier.run(query=query)

        if result[1] == "output_1":
            output = {"query":query,
                       "query_type": 'question/statement'}
        else:
            output = {"query": "what are the {} related issues and \
                        discussions?".format(query),
                      "query_type": 'statements/keyword'}
        logging.info(output)
        return output, "output_1"
    
    def run_batch(self, queries:List[str]):
        """
        running multiple queries in one go, howeevr need the queries to be passed
        as list of string. Example: if you want to find relevant passages for
        water, food security, poverty then querylist = ["water", "food security",
        "poverty"] and then execute QueryCheck.run_batch(queries = querylist)

        Params
        --------
        queries: queries/statements/keywords in form of string encapsulated 
                within List

        Return
        ------
        output: dictionary, with key as identifier and value could be anything 
                we need to return. In this case the output contain key = 'queries'.
        
        output_1: As there is only one outgoing edge, we pass 'output_1' string
        """
        query_classifier = loadQueryClassifier()
        query_list = []
        for query in queries:
            result = query_classifier.run(query=query)
            if result[1] == "output_1":
                query_list.append(query)
            else:
                query_list.append("what are the {} related issues and \
                    discussions?".format(query))
        output = {'queries':query_list}
        logging.info(output)
        return output, "output_1"


@st.cache(allow_output_mutation=True)
def runSemanticPreprocessingPipeline(file_path:str, file_name:str, 
                split_by: Literal["sentence", "word"] = 'sentence',
                split_length:int = 2, split_overlap:int = 0,
                split_respect_sentence_boundary:bool = False,
                remove_punc:bool = False)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline.

    Params
    ------------

    file_name: filename, in case of streamlit application use 
            st.session_state['filename']
    file_path: filepath, in case of streamlit application use 
            st.session_state['filepath']
    split_by: document splitting strategy either as word or sentence
    split_length: when synthetically creating the paragrpahs from document,
            it defines the length of paragraph.
    split_overlap: Number of words or sentences that overlap when creating the 
            paragraphs. This is done as one sentence or 'some words' make sense
            when  read in together with others. Therefore the overlap is used.
    split_respect_sentence_boundary: Used when using 'word' strategy for 
            splititng of text.
    remove_punc: to remove all Punctuation including ',' and '.' or not

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
                                "UdfPreProcessor": {"remove_punc": remove_punc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap,
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})

    return output_semantic_pre


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},
                                        allow_output_mutation=True)
def loadRetriever(embedding_model:Text=None, embedding_model_format:Text = None, 
                 embedding_layer:int = None,  retriever_top_k:int = 10, 
                 max_seq_len:int=512, document_store:InMemoryDocumentStore=None):
    """
    Returns the Retriever model based on params provided.
    1. https://docs.haystack.deepset.ai/docs/retriever#embedding-retrieval-recommended
    2. https://www.sbert.net/examples/applications/semantic-search/README.html
    3. https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py

    
    Params
    ---------
    embedding_model: Name of the model to be used for embedding. Check the links
            provided in documentation
    embedding_model_format: check the github link of Haystack provided in 
            documentation embedding_layer: check the github link of Haystack 
            provided in documentation retriever_top_k: Number of Top results to
            be returned by 
    retriever max_seq_len: everymodel has max seq len it can handle, check in 
            model card. Needed to hanlde the edge cases.
    document_store: InMemoryDocumentStore, write haystack Document list to 
            DocumentStore and pass the same to function call. Can be done using 
            createDocumentStore from utils.
    
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
                model_format=embedding_model_format, use_gpu = True, 
                max_seq_len = max_seq_len )
    if check_streamlit:
        st.session_state['retriever'] = retriever
    return retriever

@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},
                    allow_output_mutation=True)
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
            retiever automatically, therefore set this value as per the model card.
    
    Return
    -------
    document_store: InMemory Document Store object type.
    
    """
    document_store = InMemoryDocumentStore(similarity = similarity, 
                                        embedding_dim = embedding_dim )
    document_store.write_documents(documents)
    
    return document_store


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None},
                                        allow_output_mutation=True)
def semanticSearchPipeline(documents:List[Document], embedding_model:Text =  None, 
                embedding_model_format:Text = None,embedding_layer:int = None,
                embedding_dim:int = 768,retriever_top_k:int = 10,
                reader_model:str =  None, reader_top_k:int = 10,
                max_seq_len:int =512,useQueryCheck = True,
                top_k_per_candidate:int = 1):
    """
    creates the semantic search pipeline and document Store object from the
    list of haystack documents. The top_k for the Reader and Retirever are kept  
    same, so that all the results returned by Retriever are used, however the 
    context is extracted by Reader for each retrieved result. The querycheck is 
    added as node to process the query. This pipeline is suited for keyword search,
    and to some extent extractive QA purpose. The purpose of Reader is strictly to
    highlight the context for retrieved result and not for QA, however as stated
    it can work for QA too in limited sense.
    There are 4 variants of pipeline it can return
    1.QueryCheck > Retriever > Reader
    2.Retriever > Reader
    3.QueryCheck > Retriever > Docs2Answers : If reader is None, 
    then Doc2answer is used to keep the output of pipeline structurally same.
    4.Retriever > Docs2Answers 

    Links

    1. https://docs.haystack.deepset.ai/docs/retriever#embedding-retrieval-recommended
    2. https://www.sbert.net/examples/applications/semantic-search/README.html
    3. https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/retriever/dense.py
    4. https://docs.haystack.deepset.ai/docs/reader


    Params
    ----------
    documents: list of Haystack Documents, returned by preprocessig pipeline.
    embedding_model: Name of the model to be used for embedding. Check the links
            provided in documentation
    embedding_model_format: check the github link of Haystack provided in 
            documentation
    embedding_layer: check the github link of Haystack provided in documentation
    embedding_dim: Document store has default value of embedding size = 768, and
            update_embeddings method of Docstore cannot infer the embedding size of 
            retiever automatically, therefore set this value as per the model card.
    retriever_top_k: Number of Top results to be returned by retriever
    reader_model: Name of the model to be used for Reader node in hasyatck 
            Pipeline. Check the links provided in documentation
    reader_top_k: Reader will use retrieved results to further find better matches.
            As purpose here is to use reader to extract context, the value is
            same as retriever_top_k.
    max_seq_len:everymodel has max seq len it can handle, check in model card. 
            Needed to hanlde the edge cases
    useQueryCheck: Whether to use the querycheck which modifies the query or not.
    top_k_per_candidate:How many answers to extract for each candidate doc 
            that is coming from the retriever

    Return
    ---------
    semanticsearch_pipeline: Haystack Pipeline object, with all the necessary 
            nodes [QueryCheck, Retriever, Reader/Docs2Answer]. If reader is None, 
            then Doc2answer is used to keep the output of pipeline structurally 
            same.

    document_store: As retriever can work only with Haystack Document Store, the
            list of document returned by preprocessing pipeline are fed into to
            get InMemmoryDocumentStore object type, with retriever updating the 
            embeddings of each paragraph in document store.

    """
    document_store = createDocumentStore(documents=documents, 
                                    embedding_dim=embedding_dim)                  
    retriever = loadRetriever(embedding_model = embedding_model,
                    embedding_model_format=embedding_model_format,
                    embedding_layer=embedding_layer,  
                    retriever_top_k= retriever_top_k, 
                    document_store = document_store,
                    max_seq_len=max_seq_len)           
    document_store.update_embeddings(retriever)
    semantic_search_pipeline = Pipeline()
    if useQueryCheck and reader_model:
        querycheck = QueryCheck()
        reader = FARMReader(model_name_or_path=reader_model,
                    top_k = reader_top_k, use_gpu=True,
                    top_k_per_candidate = top_k_per_candidate)
        semantic_search_pipeline.add_node(component = querycheck, 
                    name = "QueryCheck",inputs = ["Query"])
        semantic_search_pipeline.add_node(component = retriever, 
                    name = "EmbeddingRetriever",inputs = ["QueryCheck.output_1"])
        semantic_search_pipeline.add_node(component = reader, name = "FARMReader",
                                        inputs= ["EmbeddingRetriever"])

    elif reader_model :
        reader = FARMReader(model_name_or_path=reader_model,
                    top_k = reader_top_k, use_gpu=True,
                    top_k_per_candidate = top_k_per_candidate)
        semantic_search_pipeline.add_node(component = retriever, 
                    name = "EmbeddingRetriever",inputs = ["Query"])
        semantic_search_pipeline.add_node(component = reader,
                    name = "FARMReader",inputs= ["EmbeddingRetriever"])
    elif useQueryCheck and not reader_model:
        querycheck = QueryCheck()
        docs2answers = Docs2Answers() 
        semantic_search_pipeline.add_node(component = querycheck,
                        name = "QueryCheck",inputs = ["Query"])
        semantic_search_pipeline.add_node(component = retriever,
                        name = "EmbeddingRetriever",inputs = ["QueryCheck.output_1"])
        semantic_search_pipeline.add_node(component = docs2answers,
                        name = "Docs2Answers",inputs= ["EmbeddingRetriever"])
    elif not useQueryCheck and not reader_model:
        docs2answers = Docs2Answers()
        semantic_search_pipeline.add_node(component = retriever, 
                        name = "EmbeddingRetriever",inputs = ["Query"])
        semantic_search_pipeline.add_node(component = docs2answers, 
                        name = "Docs2Answers",inputs= ["EmbeddingRetriever"])            
        
    logging.info(semantic_search_pipeline.components)
    return semantic_search_pipeline, document_store

def runSemanticPipeline(pipeline:Pipeline, queries:Union[list,str])->dict:
    """
    will use the haystack run or run_batch based on if single query is passed 
    as string or multiple queries as List[str]
    
    Params
    -------
    pipeline: haystack pipeline, this is same as returned by semanticSearchPipeline
            from utils.semanticsearch

    queries: Either a single query or list of queries.

    Return
    -------
    results: Dict containing answers and documents as key and their respective 
            values

    """

    if type(queries) == list:
        results = pipeline.run_batch(queries=queries)
    elif type(queries) == str:
        results = pipeline.run(query=queries)
    else:
        logging.info("Please check the input type for the queries")
        return

    return results

def process_query_output(results:dict)->pd.DataFrame:
    """
    Returns the dataframe with necessary information like including
    ['query','answer','answer_offset','context_offset','context','content',
    'reader_score','retriever_score','id',]. This is designed for output given 
    by semantic search pipeline with single query and final node as reader.
    The output of pipeline having Docs2Answers as final node or multiple queries
    need to be handled separately. In these other cases, use process_semantic_output
    from utils.semantic_search which uses this function internally to make one
    combined dataframe.
    
    Params
    ---------
    results: this dictionary should have key,values with 
            keys = [query,answers,documents], however answers is optional.
            in case of [Doc2Answers as final node], process_semantic_output 
            doesnt return answers thereby setting all values contained in 
            answers to 'None'
        
    Return
    --------
    df: dataframe with all the columns mentioned in function description.
    
    """
    query_text = results['query']
    if 'answers' in results.keys():
        answer_dict = {}

        for answer in results['answers']:
            answer_dict[answer.document_id] = answer.to_dict()
    else:
        answer_dict = {}
    docs = results['documents']
    df = pd.DataFrame(columns=['query','answer','answer_offset','context_offset',
                            'context','content','reader_score','retriever_score',
                            'id'])
    for doc in docs:
        row_list = {}
        row_list['query'] = query_text
        row_list['retriever_score'] = doc.score
        row_list['id'] = doc.id
        row_list['content'] = doc.content
        if doc.id in answer_dict.keys():
            row_list['answer'] = answer_dict[doc.id]['answer']
            row_list['context'] = answer_dict[doc.id]['context']
            row_list['reader_score'] = answer_dict[doc.id]['score']
            answer_offset = answer_dict[doc.id]['offsets_in_document'][0]
            row_list['answer_offset'] = [answer_offset['start'],answer_offset['end']]
            start_idx = doc.content.find(row_list['context'])
            end_idx = start_idx + len(row_list['context'])
            row_list['context_offset'] = [start_idx, end_idx]
        else:
            row_list['answer'] = None
            row_list['context'] = None
            row_list['reader_score'] = None
            row_list['answer_offset'] = None
            row_list['context_offset'] = None
        df_dictionary = pd.DataFrame([row_list])
        df = pd.concat([df, df_dictionary], ignore_index=True)
    
    return df

def process_semantic_output(results):
    """
    Returns the dataframe with necessary information like including
    ['query','answer','answer_offset','context_offset','context','content',
    'reader_score','retriever_score','id',]. Distingushes if its single query or
    multi queries by reading the pipeline output dictionary keys.
    Uses the process_query_output to get the dataframe for each query and create
    one concataneted dataframe. In case of Docs2Answers as final node, deletes 
    the answers part. See documentations of process_query_output.
    
    Params
    ---------
    results: raw output of runSemanticPipeline. 
        
    Return
    --------
    df: dataframe with all the columns mentioned in function description.
    
    """
    output = {}
    if 'query' in results.keys():
        output['query'] = results['query']
        output['documents'] = results['documents']
        if results['node_id'] == 'Docs2Answers':
            pass
        else:
            output['answers'] = results['answers']
        df = process_query_output(output)
        return df
    if 'queries' in results.keys():
        df = pd.DataFrame(columns=['query','answer','answer_offset',
                                   'context_offset','context','content',
                                   'reader_score','retriever_score','id'])
        for query,answers,documents in zip(results['queries'],
                    results['answers'],results['documents']):
            output = {}
            output['query'] = query
            output['documents'] = documents
            if results['node_id'] == 'Docs2Answers':
                    pass
            else:
                output['answers'] = answers
            
            temp = process_query_output(output)
            df = pd.concat([df, temp], ignore_index=True)

            
    return df

def semanticsearchAnnotator(matches:List[List[int]], document:Text):
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
    

def semantic_keywordsearch(query:Text,documents:List[Document],
                embedding_model:Text, 
                embedding_model_format:Text, 
                embedding_layer:int,  reader_model:str,
                retriever_top_k:int = 10, reader_top_k:int = 10,
                return_results:bool = False, embedding_dim:int = 768,
                max_seq_len:int = 512,top_k_per_candidate:int =1,
                sort_by:Literal["retriever", "reader"] = 'retriever'):
    """
    Performs the Semantic search on the List of haystack documents which is 
    returned by preprocessing Pipeline.

    Params
    -------
    query: Keywords that need to be searche in documents.
    documents: List fo Haystack documents returned by preprocessing pipeline.
    
    """
    semanticsearch_pipeline, doc_store = semanticSearchPipeline(documents = documents,
                        embedding_model= embedding_model, 
                        embedding_layer= embedding_layer,
                        embedding_model_format= embedding_model_format,
                        reader_model= reader_model, retriever_top_k= retriever_top_k,
                        reader_top_k= reader_top_k, embedding_dim=embedding_dim,
                        max_seq_len=max_seq_len,
                        top_k_per_candidate=top_k_per_candidate)

    raw_output = runSemanticPipeline(semanticsearch_pipeline,query)
    results_df = process_semantic_output(raw_output)
    if sort_by == 'retriever':
        results_df = results_df.sort_values(by=['retriever_score'], ascending=False)
    else:
        results_df = results_df.sort_values(by=['reader_score'], ascending=False)

    if return_results:
        return results_df
    else:
        if check_streamlit:
            st.markdown("##### Top few semantic search results #####")
        else:
            print("Top few semantic search results")
        for i in range(len(results_df)):
            if check_streamlit:
                st.write("Result {}".format(i+1))
            else:
                print("Result {}".format(i+1))
            semanticsearchAnnotator([results_df.loc[i]['context_offset']],
                        results_df.loc[i]['content'] )