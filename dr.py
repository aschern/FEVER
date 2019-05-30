import warnings
warnings.filterwarnings("ignore")

from allennlp.predictors.predictor import Predictor
import ast
from mediawiki import MediaWiki
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pandas as pd
import pickle
from unidecode import unidecode


data_path = os.path.dirname(os.path.abspath(''))

wiki = MediaWiki()
ps = PorterStemmer()
constituency_parser = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
dependency_parser = Predictor.from_path('https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz')
#ner_parser = Predictor.from_path('https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz')


def extract_noun_phrase_from_tree(node):
    '''
    finds noun phareses from the parsing tree
    '''
    if node['nodeType'] == 'NP':
        yield node['word']
    for child in node.get('children', []):
        yield from extract_noun_phrase_from_tree(child)


def extract_titles(search, urls, stem_claim=None, stem=True):
    '''
    Args:
    search - search phrase
    urls - current Wikipedia urls set
    stem_claim
    stem - stemming usage for results filtering
    
    Returns:
    Wikipedia urls
    '''
    assert not stem or stem_claim is not None
    
    for p in wiki.search(search, results=3):    # top-7,5,2 works worse
        title = p.split('(')[0]
        if title == 'Trollhunter':
            title = 'Trollhunters'
        
        in_claim = True    # results filtering
        if stem:
            stem_title = [unidecode(ps.stem(word)) for word in word_tokenize(title)]
            for word in stem_title:
                if word not in stem_claim:
                    in_claim = False
        else:
            for word in title:
                if word not in claim:
                    in_claim = False
        if in_claim:
            urls.add(unidecode(p.replace(' ', '_')))
    return urls


def find_documents(claim, stem=True):
    '''
    compiles and makes search queries
    
    Args:
    claim
    stem - stemming usage for results filtering
    
    Returns:
    Wikipedia urls
    '''
    if stem:
        stem_claim = [unidecode(ps.stem(word)) for word in word_tokenize(claim)]
    
    res = constituency_parser.predict(sentence=claim)
    urls = set()
    for noun_phrase in extract_noun_phrase_from_tree(res['hierplane_tree']['root']):
        urls = extract_titles(noun_phrase, urls, stem_claim, stem)
    
    word = dependency_parser.predict(sentence=claim)['hierplane_tree']['root']['word']
    prev = claim.split(word)[0]
    if prev != '':
        urls = extract_titles(prev, urls, stem_claim, stem)
    
    # named entities recognition module usage
    #res = ner_parser.predict(sentence=claim)
    #ner = ' '.join(np.array(res['words'])[np.array(res['tags']) != 'O'])
    #if ner != '':
    #    urls = extract_titles(ner, urls, stem_claim, stem)
    
    return urls


def document_retrieval(task_type, claims=None):
    '''
    Args:
    task_type - "train", "eval", "test" or "demo"
    claims - list of claims for "demo" task type
    
    Returns:
    Document Retrieval stage results
    '''  
    assert task_type in ['test', 'train', 'eval', 'demo'], "the 'task_type' parameter must take one of four values: 'test', 'train', 'eval' or 'demo'"
    assert task_type != 'demo' or claims is not None, "claims list should be provided in the 'demo' task type"
    
    if task_type != 'demo':
        if task_type == 'train':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/train.csv"))
        if task_type == 'eval':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
        if task_type == 'test':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_test.csv"))
        claims = train.claim.values
        verifiable = train.verifiable.values
    
    documents = {}
    length = len(claims)
    for i in range(length):
        if task_type != 'train' or verifiable[i] != 'VERIFIABLE':
            claim = claims[i]
            urls = find_documents(claim)
            documents[i] = tuple(urls)

    with open('results/documents_{}.pickle'.format(task_type), 'wb') as f:
        pickle.dump(documents, f)


def dr_quality_estimation():
    '''
    Returns:
    Validation score for document retrieval stage
    '''
    train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    
    claims = train.claim.values
    evidences = train.evidence.values
    labels = train.label.values
    verifiable = train.verifiable.values
    
    with open('results/documents_dev.pickle', 'rb') as f:
        documents = pickle.load(f)
    total = len(documents)
    
    k = 0
    for i in range(total):
        found = True
        if verifiable[i] == 'VERIFIABLE':
            urls = documents[i]
            for p in evidences[i]:
                if len(p) == 1:
                    found = False
                    if unidecode(p[0][2].replace('-LRB-', '(').replace('-RRB-', ')').replace('-COLON-', ':')) in urls:
                        found = True
                        break
        if found:
            k += 1
    return k / total
  