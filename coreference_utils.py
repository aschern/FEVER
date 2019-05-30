import logging
from nltk.corpus import stopwords
import numpy as np
import pickle
import re
import os

logging.getLogger().setLevel(logging.INFO)

data_path = os.path.dirname(os.path.abspath(''))
stop = stopwords.words('english')
stop.append("'s")

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("\t"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text) 
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = text.replace("vs.","vs")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def stanford_data_adapter(data):
    sents = []
    for sent in data['sentences']:
        sents.append([])
        for token in sent['tokens']:
            sents[-1].append(token['originalText'])

    clusters = []
    if data['corefs'] is not None:
        for num, mentions in data['corefs'].items():
            clusters.append([])
            for mention in mentions:
                start = np.cumsum([0]+list(map(len, sents)))[mention['sentNum']-1] + mention['startIndex']-1
                end = np.cumsum([0]+list(map(len, sents)))[mention['sentNum']-1] + mention['endIndex']-2
                clusters[-1].append([start, end])
            
    return sum(sents, []), clusters


def allen_data_adapter(data):
    return data['document'], data['clusters']


def clusters_repr(tokens, clusters):
    cor_clusters = dict()
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            mentions = set()
            for span in cluster:
                mention = [tok for tok in tokens[span[0]:span[1]+1] if tok.lower() not in stop and tok.lower() not in string.punctuation]
                if len(mention) > 0:
                    mentions.add(' '.join(mention).lower())
            copy_ment = mentions.copy()
            for mention in copy_ment:
                mentions.discard(mention)
                if mention not in ' # '.join(list(mentions)):
                    mentions.add(mention)
            if len(mentions) > 0:
                cor_clusters[i] = ' # '.join(list(mentions))
    return cor_clusters


def coref_in_start_sents(data, title, ann_type):
    if ann_type == 'Stanford':
        tokens, clusters = stanford_data_adapter(data)
    else:
        tokens, clusters = allen_data_adapter(data)
    clusters_representation = clusters_repr(tokens, clusters)
    sent_ends = np.where(np.array(tokens) == '.')[0]
    cluster_correction = dict()
    for i in clusters_representation:
        for span in clusters[i]:
            mention = [tok for tok in tokens[span[0]:span[1]+1] if tok.lower() not in stop and tok.lower() not in string.punctuation]
            if len(mention) == 0:  # == only for stop-words; >= for all
                sent_start = np.where(span[0] < sent_ends)[0][0]
                cluster_correction.setdefault(sent_start, set())
                cluster_correction[sent_start].add(i)
                
    sents = split_into_sentences(w[title])
    for i in cluster_correction:
        for j in cluster_correction[i]:
            #addition = [el for el in clusters_representation[j].split(' # ') if el not in sents[i].lower()]
            #if len(addition) != 0:
            #    sents[i] = ' # '.join(addition) + ' # ' + sents[i]
            sents[i] = clusters_representation[j] + ' # ' + sents[i]
    
    return sents


def coref_in_start(title, sents):  
    prev = w[title].replace('  ', '  .  .').split(' . ')
    coref_text = ' '.join(sents).replace('  ', '  .  .').split(' . ')

    new_text = []
    k = 0
    for el in prev:
        if el == '':
            new_text.append('')
        else:
            new_text.append(coref_text[k])
            k += 1
    return new_text


def read_corefs(ann_type):
    '''
    ann_type - Stanford or AllenNLP
    
    Returns: corefs[id][title]
    '''
    logging.info('loading {} coreference'.format(ann_type))
    if ann_type == 'Stanford':
        with open(os.path.join(data_path, 'coreferences_dev_stanford.pickle'), 'rb') as f:
            corefs = pickle.load(f)
    if ann_type == 'Allen':
        with open(os.path.join(data_path, 'coreferences_dev.pickle'), 'rb') as f:
            corefs = pickle.load(f)
    logging.info('done')
    return corefs
