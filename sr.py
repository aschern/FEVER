import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=DeprecationWarning)

import ast
from coreference_utils import *
import csv
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import io
import logging
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import numpy as np
import os
import pandas as pd
import pickle
import random
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from tqdm import tqdm_notebook as tqdm
from unidecode import unidecode


logging.getLogger().setLevel(logging.INFO)

data_path = os.path.dirname(os.path.abspath(''))
wiki_path = os.path.join(data_path, "FEVER_data/wiki_pages.csv")
glove_path = os.path.join(data_path, 'glove.840B.300d.txt')
fasttext_path = os.path.join(data_path, 'cc.en.300.vec')

tknzr = TweetTokenizer()
stop = stopwords.words('english')

w2v_embs = {}
w = {}
w_decoded = {}

# the best hyper parameters are fixed
vectorizer = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, max_df=0.85, binary=True)


def load_fasttext(fname='cc.en.300.vec'):
    global w2v_embs
    logging.info('loading fasttext, it will take around 50 seconds')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    i = 0
    for line in fin:
        if i > 500000:
            break
        tokens = line.rstrip().split(' ')
        w2v_embs[tokens[0]] = np.array(list(map(float, tokens[1:])))
        i += 1
    logging.info('done')


def load_glove(fname='glove.840B.300d.txt'):
    global w2v_embs
    logging.info('loading glove, it will take around 50 seconds')
    i = 0
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        for vec in f:
            i += 1
            if i > 500000:
                break
            try:
                line = vec.split()
                w2v_embs[line[0]] = np.array([float(val) for val in line[1:]])
            except:
                continue
    logging.info('done')


def w2v(word):
    return w2v_embs.get(word, np.zeros(300))


def load_wiki(path="FEVER_data/wiki_pages.csv"):
    global w
    global w_decoded
    logging.info('loading Wikipedia dump')
    wiki = pd.read_csv(path)  #nrows=70000
    w = dict()
    w_decoded = dict()
    for r in wiki.iterrows():
        if r[1]['id'] == r[1]['id']:
            w[unidecode(r[1]['id'])] = r[1]['text']
            w_decoded[unidecode(r[1]['id'])] = r[1]['id']
    del wiki
    logging.info('done')


def delete_stopwords(sentence):
    return ' '.join([x.lower() for x in sentence.split(' ') if x.lower() not in stop and x.lower() not in string.punctuation])


def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def find_sentences_in_document(mode, data, claim, title,
                               claim_emb=None, word_vectors=None, claim_entities=None, argsort=False):
    '''
    mode 1: tf-idf
    mode 2: jaccard
    mode 3: glove
    mode 4: fasttext
    mode 5: 0.5 * tf-idf + 0.5 * glove
    mode 6: wmd
    mode 7: entity filtering
    mode 8: bm25
    '''
    
    if data != []:
        sents = coref_in_start_sents(data, title)
        text = coref_in_start(title, sents)
    else:
        text = w[unidecode(title)].replace('  ', '  .  .').split(' . ')

    if mode == 1:
        tfidf = vectorizer.fit_transform([claim] + text)
        sims = cosine_similarity(tfidf[0], tfidf).flatten()[1:]

    elif mode == 2:
        jaccard_similarities = []
        for sent in text:
            jaccard_similarities.append(get_jaccard_sim(claim[:-1], delete_stopwords(sent)[:-1]))
        sims = np.array(jaccard_similarities)

    elif mode == 3 or mode == 4:
        sent_embs = []
        for sent in text:
            if len(sent) != 0:
                sent_embs.append(np.mean([w2v_embs[word] for word in tknzr.tokenize(sent)
                                         if word in w2v_embs and word not in stop and word not in string.punctuation], axis=0))
            else:
                sent_embs.append(np.zeros(300))
        sent_embs = np.array(sent_embs)
        sims = cosine_similarity([claim_emb], sent_embs).flatten()

    elif mode == 5:
        sent_embs = []
        tfidf = vectorizer.fit_transform([claim] + text)
        tfidf_similarities = cosine_similarity(tfidf[0], tfidf).flatten()[1:]
        for sent in text:
            if len(sent) != 0:
                sent_embs.append(np.mean([w2v_embs[word] for word in tknzr.tokenize(sent)
                                         if word in w2v_embs and word not in stop and word not in string.punctuation], axis=0))
            else:
                sent_embs.append(np.zeros(300))
        sent_embs = np.array(sent_embs)
        cosine_similarities = cosine_similarity([claim_emb], sent_embs).flatten()
        sims = 0.5 * tfidf_similarities + 0.5 * cosine_similarities
    
    elif mode == 6:
        sims = []
        for sent in text:
            sims.append(word_vectors.wmdistance(tknzr.tokenize(claim), tknzr.tokenize(sent)))
        sims = np.array(sims)
        
    elif mode == 7:
        coref_text = text
        tfidf = vectorizer.fit_transform([claim] + text)
        sims = cosine_similarity(tfidf[0], tfidf).flatten()[1:]
        for i in range(len(coref_text)):
            entity_found = False
            for entity in claim_entities:
                if entity.lower() in coref_text[i].lower():
                    entity_found = True
                    break
            sims[i] += float(entity_found) * 0.3
    
    elif mode == 8:
        corpus = [[ks.stem(word.lower()) for word in tknzr.tokenize(sent) 
                  if word not in stop and len(word) > 2 and word not in string.punctuation] for sent in text]
        '''
        # document stop-words filtering
        word_frequencies = [Counter(document) for document in corpus]
        document_frequencies = Counter()
        for word_frequency in word_frequencies:
            document_frequencies.update(word_frequency.keys())
        new_stop = {k for k, v in document_frequencies.items() if v > len(corpus) * 0.85}
        corpus = [[word for word in sent if word not in new_stop] for sent in corpus]
        '''
        
        bm25 = BM25L(corpus) 
        #bm25 = BM25L(corpus, b=0.01, k1=1, delta=0.99)
        #bm25 = BM25Okapi(corpus)
        #bm25 = BM25Plus(corpus)
        tokenized_query = [ks.stem(word.lower()) for word in tknzr.tokenize(claim) 
                          if word not in stop and len(word) > 2 and word not in string.punctuation]
        sims = bm25.get_scores(tokenized_query)
        
    if argsort:
        return sims.argsort()
    else:
        return sims


def find_sentences(urls, coref, claim, mode, top, word_vectors=None):
    result = {}
    if mode in [3, 4, 5]:
        claim_emb = np.mean([w2v_embs[word] for word in tknzr.tokenize(claim) 
                             if word in w2v_embs and word not in stop and word not in string.punctuation], axis=0)
    else:
        claim_emb = None
    
    sims = np.array([])
    lens = [0]
    titles = []
    
    for title in urls:
        title = unidecode(title.replace('(', '-LRB-').replace(')', '-RRB-').replace(':' , '-COLON-'))
        try:
            data = coref.get(title, [])
            similarities = find_sentences_in_document(mode, data, claim, title, claim_emb, word_vectors)
            sims = np.concatenate([sims, similarities])
            lens.append(len(similarities) + lens[-1])
            titles.append(title)
        except:
            continue
    
    lens = np.array(lens)
    for ind in sims.argsort()[-top:]:
        res = lens[lens - ind <= 0].argmax()
        result.setdefault(titles[res], dict())
        result[titles[res]][ind - lens[res]] = sims[ind]
    
    return result

    
def sr_quality_estimation(mode=1, top=5, use_coreference=False, ann_type='Stanford', total=None):
    '''
    mode 1: tf-idf
    mode 2: jaccard
    mode 3: glove
    mode 4: fasttext
    mode 5: 0.5 * tf-idf + 0.5 * glove
    mode 6: wmd
    mode 7: entity filtering
    mode 8: bm25
    '''
    train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    
    claims = train.claim.values
    evidences = train.evidence.values
    labels = train.label.values
    verifiable = train.verifiable.values
    
    with open(os.path.join(data_path, 'documents_dev.pickle'), 'rb') as f:
        documents = pickle.load(f)
    
    if total is None:
        total = len(claims)     # actually coreference resolution exists only for the first 7000 dev. claims
    
    coref = {}
    if use_coreference:
        corefs = read_corefs(ann_type)
    
    if w2v_embs == {}:
        if mode == 3 or mode == 5:
            load_glove(glove_path)
        if mode == 4:
            load_fasttext(fasttext_path)     
   
    word_vectors = None
    if mode == 6:
        model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
        word_vectors = model.wv
        
    if w == {}:
        load_wiki(wiki_path)
    
    k = 0
    for i in tqdm(range(total)):
        found = True
        if verifiable[i] == 'VERIFIABLE':
            claim = claims[i]
            urls = documents[i]
            if use_coreference and i < len(corefs):
                 coref = corefs[i]
            result = find_sentences(urls, coref, claim, mode, top, word_vectors)
            for p in evidences[i]:
                if len(p) == 1:
                    found = False
                    if p[0][3] in result.get(unidecode(p[0][2]), []):
                        found = True
                        break
        if found:
            k += 1
    
    return k / total * 100


def save_sr_results(claims, documents, mode, task_type, total=None, top=20, use_coreference=False, ann_type='Stanford'):
    if total is None:
        total = len(claims)
    
    if use_coreference:
        corefs = read_corefs(ann_type)
    coref = {}
    
    if w2v_embs == {}:
        if mode == 3 or mode == 5:
            load_glove(glove_path)
        if mode == 4:
            load_fasttext(fasttext_path)     
    
    word_vectors = None
    if mode == 6:
        model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
        word_vectors = model.wv
        
    if w == {}:
        load_wiki(wiki_path)
    
    saver = []
    for i in tqdm(range(total)):
        urls = []
        claim = claims[i]
        urls = documents[i]
        if use_coreference and i < len(corefs):
            coref = corefs[i]
        result = find_sentences(urls, coref, claim, mode, top, word_vectors)
        saver.append(result)
    
    with open('results/results_{}.pickle'.format(task_type), 'wb') as f:
        pickle.dump(saver, f)


def create_bert_file(task_type, claims, mode=1, total=None, top=20, 
                     use_coreference_bert=False, use_coreference_search=False, ann_type='Stanford'):
    if w == {}:
        load_wiki(wiki_path)
        
    with open('results/documents_{}.pickle'.format(task_type), 'rb') as f:
        documents = pickle.load(f)
    
    save_sr_results(claims, documents, mode, task_type, total, top, use_coreference_search, ann_type)
    with open('results/results_{}.pickle'.format(task_type), 'rb') as f:
        saver = pickle.load(f)
    
    if use_coreference_bert:
        corefs = read_corefs(ann_type)
        new_corefs = {}
        for i in range(len(corefs)):
            for title in corefs[i]:
                new_corefs[title] = corefs[i][title]
        del corefs
    
    with open('results/pred_{}.tsv'.format(task_type), "w", encoding="utf-8") as file:
        f = csv.writer(file, delimiter='\t')
        f.writerow(["text_a", "text_b", "title", "sent", "index"])
        for i in tqdm(range(len(saver))):
            new_result = {}
            result = saver[i]
            text_b = claims[i]
            
            for title in result:
                for sent in result[title]:
                    new_result[(title, sent)] = result[title][sent]
            
            for el in list(sorted(new_result.items(), key=lambda kv: -kv[1])):
                doc = el[0][0]
                t = doc.replace('_', ' ')#.split('-LRB-')[0]
                sent = el[0][1]
                text = w[doc].replace('  ', '  .  .').split(' . ')
                
                if use_coreference_bert:
                    try:
                        data = new_corefs[doc]
                        sents = coref_in_start_sents(data, doc, ann_type)
                        text = coref_in_start(doc, sents)
                    except:
                        pass
                
                if text[sent] != '' and text[sent].strip()[0] == '.':
                    text[sent] = text[sent].strip()[1:]
                text_a = '# ' + t + ' # ' + text[sent] + ' . '
                f.writerow([text_a, text_b, doc, sent, i])

                
def sentence_retrieval(task_type, claims=None, mode=1, total=None, top=20, 
                       use_coreference_bert=False, use_coreference_search=False, ann_type='Stanford'):
    '''
    Args:
    task_type - "eval", "test" or "demo"
    claims - list of claims for "demo" task type
    mode - type of retirieval (see find_sentences_in_document), 1 (tf-idf) as default
    total - the number of claims used for prediction (None as default -> all claims)
    top - the number of evidence sentences for each claim (20 as default)
    use_coreference_bert - coreference usage for bert file creating
    use_coreference_bert - coreference usage for searching
    ann_type - the type of coreference annotator (Allen or Stanford)
    
    Returns:
    bert file "results/pred_{task_type}.tsv" for further prediction
    '''
    assert task_type in ['test', 'eval', 'demo'], "the 'task_type' parameter must take one of four values: 'test', 'eval' or 'demo'"
    assert task_type != 'demo' or claims is not None, "claims list should be provided in the 'demo' task type"
    
    if task_type != 'demo':
        if task_type == 'eval':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))
        if task_type == 'test':
            train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_test.csv"))
        claims = train.claim.values
    
    create_bert_file(task_type, claims, mode, total, top, use_coreference_bert, use_coreference_search, ann_type)
                
                
def create_train_bert_file(task_type, top=1):
    assert task_type in ["train", 'dev']
    
    if w == {}:
        load_wiki(wiki_path)
        
    with open('results/documents_{}.pickle'.format(task_type), 'rb') as f:
        documents = pickle.load(f)
        
    if task_type == 'train':
        train = pd.read_csv(os.path.join(data_path, "FEVER_data/train.csv"))
    if task_type == 'dev':
        train = pd.read_csv(os.path.join(data_path, "FEVER_data/shared_task_dev.csv"))    
    train['evidence'] = train['evidence'].apply(lambda x: ast.literal_eval(x))
    claims = train.claim.values
    evidences = train.evidence.values
    labels = train.label.values
    verifiable = train.verifiable.values

    with open('results/{}.tsv'.format(task_type), "w", encoding="utf-8") as file:
        f = csv.writer(file, delimiter='\t')
        f.writerow(["text_a", "text_b", "label"])
        length = len(claims)
        for i in tqdm(range(length)):
            if verifiable[i] == 'VERIFIABLE':
                for p in evidences[i]:
                    if len(p) == 1:
                        try:
                            text = w[unidecode(p[0][2])].replace('  ', '  .  .').split(' . ')
                            title = p[0][2].replace('_', ' ').split('-LRB-')[0]
                            text_a = '# ' + title + ' # ' + text[p[0][3]] + '.'
                            text_b = claims[i]
                            label = labels[i]
                            f.writerow([text_a, text_b, label])
                            break
                        except:
                            continue
            else:
                claim = claims[i]
                urls = documents[i]
                result = find_sentences(urls, {}, claim, 1, top)
                text_b = claim
                label = labels[i]
                for doc in result:
                    t = doc.replace('_', ' ').split('-LRB-')[0]
                    for sent in result[doc]:
                        text = w[unidecode(doc)].replace('  ', '  .  .').split(' . ')
                        text_a = '# ' + t + ' # ' + text[sent] + '.'
                        f.writerow([text_a, text_b, label])

    train = pd.read_csv('results/{}.tsv'.format(task_type), sep='\t')
    train = train.sample(frac=1, random_state=41).reset_index(drop=True)
    train.to_csv('results/{}.tsv'.format(task_type), sep='\t', index=False)
