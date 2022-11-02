from haystack.nodes import TfidfRetriever
from haystack.document_stores import InMemoryDocumentStore
import configparser
import spacy
import re
from spacy.matcher import Matcher
import streamlit as st
from markdown import markdown
from annotated_text import annotation

config = configparser.ConfigParser()
config.read_file(open('paramconfig.py'))


def tokenize_lexical_query(query):
    nlp = spacy.load("en_core_web_sm")    
    token_list = [token.text.lower() for token in nlp(query) if not token.is_stop]
    return token_list

def runSpacyMatcher(token_list, document):
    nlp = spacy.load("en_core_web_sm")
    spacydoc = nlp(document)
    matcher = Matcher(nlp.vocab)
    token_pattern = [[{"LOWER":token}] for token in token_list]
    matcher.add(",".join(token_list), token_pattern)
    spacymatches = matcher(spacydoc)

    matches = []
    for match_id, start, end in spacymatches:
        matches = matches + [[start, end]]
    
    return matches, spacydoc

def runRegexMatcher(token_list, document):
    matches = []
    for token in token_list:
        matches = matches + [[val.start(), val.start()+ len(token)] for val in re.finditer(token, document)]
    
    return matches, document

def searchAnnotator(matches, document):
    start = 0
    annotated_text = ""
    for match in matches:
        start_idx = match[0]
        end_idx = match[1]
        annotated_text = annotated_text + document[start:start_idx] + str(annotation(body=document[start_idx:end_idx], label="ANSWER", background="#964448", color='#ffffff'))
        start = end_idx
    
    st.write(
            markdown(annotated_text),
            unsafe_allow_html=True,
        )

def lexical_search(query,documents):

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    retriever = TfidfRetriever(document_store)
    results = retriever.retrieve(query=query, 
                            top_k= int(config.get('lexical_search','TOP_K')))
    query_tokens = tokenize_lexical_query(query)
    for result in results:
        matches, doc = runSpacyMatcher(query_tokens,result.content)
        searchAnnotator(matches, doc)

    
    
